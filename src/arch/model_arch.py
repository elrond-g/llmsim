"""
模型架构抽象层 - 定义通用的模型架构接口和基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any
import json
import os

from src.op.operator_base import (
    BaseOperator, OperatorMetadata, OperatorIO, Tensor, DataType,
    create_operator
)


class AttentionType(Enum):
    """注意力类型"""
    MHA = "mha"  # Multi-Head Attention
    MLA = "mla"  # Multi-Head Latent Attention


class ForwardMode(Enum):
    """前向传递模式"""
    EXTEND = 0  # 序列扩展模式
    DECODE = 1  # 解码模式


@dataclass
class ModelConfig:
    """模型配置基类"""
    model_type: str = ""
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 11008
    
    # 从 JSON 配置文件加载
    @staticmethod
    def from_json(config_path: str) -> "ModelConfig":
        """从 JSON 文件加载配置"""
        if not os.path.exists(config_path):
            raise RuntimeError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # 根据 model_type 选择合适的配置类
        model_type = data.get('model_type', '')
        
        if model_type == 'deepseek_v3':
            return DeepSeekV3Config.from_dict(data)
        elif model_type == 'qwen3':
            return Qwen3Config.from_dict(data)
        else:
            return ModelConfig.from_dict(data)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        config = ModelConfig()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        config.model_type = data.get('model_type', '')
        return config


@dataclass
class DeepSeekV3Config(ModelConfig):
    """DeepSeek V3 模型配置"""
    model_type: str = "deepseek_v3"
    
    # 特定参数
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    
    first_k_dense_replace: int = 3
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8
    
    attention_type: AttentionType = AttentionType.MLA
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DeepSeekV3Config":
        """从字典创建配置"""
        config = DeepSeekV3Config()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        return config


@dataclass
class Qwen3Config(ModelConfig):
    """Qwen3 模型配置"""
    model_type: str = "qwen3"
    
    # 特定参数
    head_dim: int = 128
    attention_type: AttentionType = AttentionType.MHA
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Qwen3Config":
        """从字典创建配置"""
        config = Qwen3Config()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                setattr(config, key, value)
        return config


@dataclass
class ScheduleConfig:
    """调度配置"""
    batch_size: int = 64
    max_seqlen: int = 4096
    mode: ForwardMode = ForwardMode.EXTEND
    
    # 并行化配置
    tp_size: int = 16  # Tensor Parallel
    dp_size: int = 4   # Data Parallel
    ep_size: int = 16  # Expert Parallel
    
    # 特殊功能开关
    is_mtp: bool = True  # Multi-Token Prediction
    deepep: bool = True  # Deep Expert Parallel
    enable_moe_dense_fully_dp: bool = False


class BaseModelArch(ABC):
    """模型架构基类"""
    
    def __init__(self, model_config: ModelConfig, schedule_config: ScheduleConfig):
        """
        初始化模型架构
        
        Args:
            model_config: 模型配置
            schedule_config: 调度配置
        """
        self.model_config = model_config
        self.schedule_config = schedule_config
        self.operators: List[BaseOperator] = []
        self.attention_operators: Dict[str, List[BaseOperator]] = {}
        self.transfer_operators: List[BaseOperator] = []
    
    @abstractmethod
    def build_operators(self) -> List[BaseOperator]:
        """构建模型的算子图"""
        pass
    
    def get_seq_length(self) -> int:
        """根据模式获取序列长度"""
        if self.schedule_config.mode == ForwardMode.EXTEND:
            return self.schedule_config.max_seqlen
        elif self.schedule_config.mode == ForwardMode.DECODE:
            return self.schedule_config.batch_size
        return self.schedule_config.max_seqlen
    
    def _add_operator(self, operator: BaseOperator) -> None:
        """添加算子到操作符列表"""
        self.operators.append(operator)
    
    def _add_attention_operator(self, key: str, operators: List[BaseOperator]) -> None:
        """添加注意力算子"""
        self.attention_operators[key] = operators
    
    def _add_transfer_operator(self, operator: BaseOperator) -> None:
        """添加传输算子"""
        self.transfer_operators.append(operator)


class SimpleTransformerArch(BaseModelArch):
    """简单 Transformer 模型架构（如 Qwen3）"""
    
    def build_operators(self) -> None:
        """构建标准 Transformer 算子"""
        mc = self.model_config
        sc = self.schedule_config
        
        assert mc.num_attention_heads % sc.tp_size == 0
        if mc.num_key_value_heads > sc.tp_size:
            assert mc.num_key_value_heads % sc.tp_size == 0
        else:
            assert sc.tp_size % mc.num_key_value_heads == 0
        
        # 计算每个 rank 的头数
        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        kv_heads_per_rank = max(1, mc.num_key_value_heads // sc.tp_size)
        seq_len = self.get_seq_length()
        head_dim = getattr(mc, 'head_dim', mc.hidden_size // mc.num_attention_heads)
        
        # 1. QKV 投影
        qkv_proj_metadata = OperatorMetadata(
            name='qkv_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim),
                weight_shape=Tensor(mc.hidden_size, (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator('matmul', qkv_proj_metadata))
        
        # 2. 输出投影
        o_proj_metadata = OperatorMetadata(
            name='o_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, num_heads_per_rank * head_dim),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(num_heads_per_rank * head_dim, mc.hidden_size),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator('matmul', o_proj_metadata))
        
        # 3. 注意力核心
        attn_operators = []
        
        # Q-K 注意力
        qk_metadata = OperatorMetadata(
            name='qk',
            op_type='attention',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.BF16,
                output_dtype=DataType.FP32,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=mc.num_hidden_layers,
        )
        attn_operators.append(create_operator('attention', qk_metadata))
        
        # Q-K-V 注意力
        qkv_metadata = OperatorMetadata(
            name='qkv',
            op_type='attention',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, sc.max_seqlen),
                output_shape=Tensor(seq_len, head_dim),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=kv_heads_per_rank,
            num_layers=mc.num_hidden_layers,
        )
        attn_operators.append(create_operator('attention', qkv_metadata))
        
        self._add_attention_operator('attention', attn_operators)
        
        # 4. 前馈网络 (FFN)
        assert mc.intermediate_size % sc.tp_size == 0
        intermediate_size = mc.intermediate_size // sc.tp_size
        
        # Gate-Up 投影
        gate_up_metadata = OperatorMetadata(
            name='dense_gate_up_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, 2 * intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * intermediate_size),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator('matmul', gate_up_metadata))
        
        # Down 投影
        down_metadata = OperatorMetadata(
            name='dense_down_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, intermediate_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(intermediate_size, mc.hidden_size),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=1,
            num_layers=mc.num_hidden_layers,
        )
        self._add_operator(create_operator('matmul', down_metadata))


class MixtureOfExpertsArch(BaseModelArch):
    """混合专家模型架构（如 DeepSeek V3）"""
    
    def build_operators(self) -> None:
        """构建 MoE 模型的算子"""
        mc = self.model_config
        sc = self.schedule_config
        
        # 处理 MoE 配置的特殊情况
        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError(f"MoE 架构需要 DeepSeekV3Config，但收到 {type(mc)}")
                
        # 1. 建立注意力层
        num_attn_layers = mc.num_hidden_layers + (1 if sc.is_mtp else 0)
        self._build_attention_operators(num_attn_layers)
        
        # 2. 建立密集层（前 K 层）
        self._build_dense_operators(mc.first_k_dense_replace)
        
        # 3. 建立 MoE 层
        moe_layers = mc.num_hidden_layers - mc.first_k_dense_replace + (1 if sc.is_mtp else 0)
        self._build_moe_operators(moe_layers)
        
        # 4. 建立 Deep-EP 传输算子（如果启用）
        if sc.deepep:
            self._build_deepep_operators(moe_layers)
    
    def _build_attention_operators(self, num_layers) -> None:
        # ====================
        # 1. QKV 投影算子
        # ====================
        """构建注意力算子"""
        mc = self.model_config
        sc = self.schedule_config
        
        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError("Expected DeepSeekV3Config")
        
        assert mc.num_attention_heads % sc.tp_size == 0
        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        seq_len = self.get_seq_length()
        qk_head_dim = mc.qk_nope_head_dim + mc.qk_rope_head_dim
        # QKV A 投影算子Meta信息
        q_a_kv_a_metadata = OperatorMetadata(
            name='q_a_kv_a',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, mc.q_lora_rank + mc.kv_lora_rank + mc.qk_rope_head_dim),
                weight_shape=Tensor(mc.hidden_size, mc.q_lora_rank + mc.kv_lora_rank + mc.qk_rope_head_dim),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator('matmul', q_a_kv_a_metadata))
        
        # Q B 投影算子Meta信息
        q_b_metadata = OperatorMetadata(
            name='q_b',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.q_lora_rank),
                output_shape=Tensor(seq_len, num_heads_per_rank * qk_head_dim),
                weight_shape=Tensor(mc.q_lora_rank, num_heads_per_rank * qk_head_dim),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator('matmul', q_b_metadata))
        
        # KV B 投影
        kv_b_metadata = OperatorMetadata(
            name='kv_b',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.kv_lora_rank),
                output_shape=Tensor(seq_len, num_heads_per_rank * (mc.v_head_dim + mc.qk_nope_head_dim)),
                weight_shape=Tensor(mc.kv_lora_rank, num_heads_per_rank * (mc.v_head_dim + mc.qk_nope_head_dim)),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator('matmul', kv_b_metadata))
        
        # 输出投影
        o_proj_metadata = OperatorMetadata(
            name='o_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, num_heads_per_rank * mc.v_head_dim),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(num_heads_per_rank * mc.v_head_dim, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator('matmul', o_proj_metadata))

         # 3. 注意力核心
        attn_operators = []
        qk_nope_metadata = OperatorMetadata(
            name='qk_nope',
            op_type='attention',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.qk_nope_head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(mc.qk_nope_head_dim, sc.max_seqlen),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        #self._add_operator(create_operator('matmul', qk_nope_metadata))
        attn_operators.append(create_operator('attention', qk_nope_metadata))

        qk_rope_metadata = OperatorMetadata(
            name='qk_rope',
            op_type='attention',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.qk_rope_head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(mc.qk_rope_head_dim, sc.max_seqlen),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        #self._add_operator(create_operator('matmul', qk_rope_metadata))
        attn_operators.append(create_operator('attention', qk_rope_metadata))

        qkv_metadata = OperatorMetadata(
            name='qkv',
            op_type='attention',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, sc.max_seqlen),
                output_shape=Tensor(seq_len, mc.v_head_dim),
                weight_shape=Tensor(sc.max_seqlen, mc.v_head_dim),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        #self._add_operator(create_operator('matmul', qkv_metadata))
        attn_operators.append(create_operator('attention', qkv_metadata))
        self._add_attention_operator('attention', attn_operators)
        

    
    def _build_dense_operators(self, num_dense_layers: int) -> None:
        # ====================
        # 2. Dense 层算子
        # ====================
        """构建密集层算子"""
        mc = self.model_config
        sc = self.schedule_config
        
        seq_len = self.get_seq_length()
        assert mc.intermediate_size % sc.tp_size == 0
        intermediate_size = mc.intermediate_size
        if not sc.enable_moe_dense_fully_dp:
            intermediate_size = intermediate_size // sc.tp_size
        
        # Gate-Up 投影
        gate_up_metadata = OperatorMetadata(
            name='dense_gate_up_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, 2 * intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_dense_layers,
        )
        self._add_operator(create_operator('matmul', gate_up_metadata))
        
        # Down 投影
        down_metadata = OperatorMetadata(
            name='dense_down_proj',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, intermediate_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_dense_layers,
        )
        self._add_operator(create_operator('matmul', down_metadata))
    
    def _build_moe_operators(self, num_moe_layers: int) -> None:
        # ====================
        # 3. MoE 层算子
        # ====================
        """构建 MoE 算子"""
        mc = self.model_config
        sc = self.schedule_config
        
        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError("Expected DeepSeekV3Config")
        
        seq_len = self.get_seq_length()
        assert mc.n_routed_experts % sc.ep_size == 0
        experts_per_rank = mc.n_routed_experts // sc.ep_size
        
        
        assert seq_len // sc.tp_size * mc.num_experts_per_tok % experts_per_rank == 0
        # 计算每个 rank 的 token 数量
        if sc.mode == ForwardMode.EXTEND:
            L_per_rank = seq_len // sc.tp_size * mc.num_experts_per_tok // experts_per_rank
        else:  # DECODE
            L_per_rank = 1  # 解码时为单 token
        
        # MoE 共享中间层大小
        _moe_intermediate_size = mc.moe_intermediate_size
        if not sc.deepep:
            _moe_intermediate_size = _moe_intermediate_size // sc.tp_size
        
        # MoE Gate 投影
        moe_gate_metadata = OperatorMetadata(
            name='moe_gate',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, mc.n_routed_experts),
                weight_shape=Tensor(mc.hidden_size, mc.n_routed_experts),
                input_dtype=DataType.FP32,
                output_dtype=DataType.FP32,
                weight_dtype=DataType.FP32,
            ),
            batch_size=1,
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator('matmul', moe_gate_metadata))
        
        # MoE Up 投影
        moe_up_metadata = OperatorMetadata(
            name='moe_up',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.hidden_size),
                output_shape=Tensor(L_per_rank, 2 * mc.moe_intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * mc.moe_intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator('matmul', moe_up_metadata))
        
        # MoE Down 投影
        moe_down_metadata = OperatorMetadata(
            name='moe_down',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.moe_intermediate_size),
                output_shape=Tensor(L_per_rank, mc.hidden_size),
                weight_shape=Tensor(mc.moe_intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator('matmul', moe_down_metadata))
        
        # 共享专家 Up 投影
        share_up_metadata = OperatorMetadata(
            name='share_up',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, 2 * _moe_intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * _moe_intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator('matmul', share_up_metadata))
        
        # 共享专家 Down 投影
        share_down_metadata = OperatorMetadata(
            name='share_down',
            op_type='matmul',
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, _moe_intermediate_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(_moe_intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator('matmul', share_down_metadata))
    
    def _build_deepep_operators(self, num_moe_layers: int) -> None:
        # ====================
        # 4. Deep-EP 传输算子
        # ====================
        """构建 Deep-EP 传输算子"""
        mc = self.model_config
        sc = self.schedule_config
        
        if not isinstance(mc, DeepSeekV3Config):
            raise ValueError("Expected DeepSeekV3Config")
        
        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen // sc.tp_size
            dispatch_bandwidth = 85.0  # GB/s
            combine_bandwidth = 85.0
        else:  # DECODE
            L = sc.batch_size // sc.tp_size
            dispatch_bandwidth = 18.58  # GB/s
            combine_bandwidth = 22.64


        # 分发传输, prefill 或者 decode 下面的逻辑一致，只是dispatch_bandwidth 和 combine_bandwidth 的值不同
        dispatch_metadata = OperatorMetadata(
            name='dispatch',
            op_type='transfer',
            io_config=OperatorIO(
                input_shape=Tensor(L, mc.hidden_size),
                output_shape=Tensor(L, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.INT8,
            ),
            batch_size=mc.num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        dispatch_op = create_operator('transfer', dispatch_metadata)
        # 设置带宽（用于传输时间计算）
        dispatch_op._bandwidth_gb_s = dispatch_bandwidth
        self._add_transfer_operator(dispatch_op)
        
        # 合并传输
        combine_metadata = OperatorMetadata(
            name='combine',
            op_type='transfer',
            io_config=OperatorIO(
                input_shape=Tensor(L, mc.hidden_size),
                output_shape=Tensor(L, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.BF16,
            ),
            batch_size=mc.num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        combine_op = create_operator('transfer', combine_metadata)
        # 设置带宽（用于传输时间计算）
        combine_op._bandwidth_gb_s = combine_bandwidth
        self._add_transfer_operator(combine_op)


def create_model_arch(model_config: ModelConfig, schedule_config: ScheduleConfig) -> BaseModelArch:
    """
    工厂函数 - 创建合适的模型架构
    
    Args:
        model_config: 模型配置
        schedule_config: 调度配置
        
    Returns:
        模型架构实例
    """
    model_type = model_config.model_type.lower()
    
    if model_type in ('deepseek_v3', 'qwen3_moe'):
        return MixtureOfExpertsArch(model_config, schedule_config)
    elif model_type == 'qwen3':
        return SimpleTransformerArch(model_config, schedule_config)
    else:
        # 默认使用标准 Transformer 架构
        return SimpleTransformerArch(model_config, schedule_config)