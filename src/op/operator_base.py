"""
算子抽象层 - 提供通用的算子定义和接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class DataType(Enum):
    """数据类型"""
    INT8 = 1
    FP16 = 2
    BF16 = 2
    FP32 = 4
    FP64 = 8


@dataclass
class Tensor:
    """张量定义"""
    m: int = 0
    n: int = 0
    
    def size(self) -> int:
        """计算张量元素数量"""
        return self.m * self.n


@dataclass
class OperatorIO:
    """算子输入输出定义"""
    input_shape: Tensor = field(default_factory=Tensor)
    output_shape: Tensor = field(default_factory=Tensor)
    weight_shape: Tensor = field(default_factory=Tensor)
    
    input_dtype: DataType = DataType.BF16
    output_dtype: DataType = DataType.BF16
    weight_dtype: DataType = DataType.BF16


@dataclass
class OperatorMetadata:
    """算子元数据"""
    name: str = ""
    op_type: str = ""  # 算子类型: matmul, conv, etc
    description: str = ""
    
    io_config: OperatorIO = field(default_factory=OperatorIO)
    
    # 执行参数
    batch_size: int = 1
    num_layers: int = 1
    
    # 并行化参数
    parallelization_dim: Optional[str] = None  # 并行维度标记


class BaseOperator(ABC):
    """基础算子抽象类"""
    
    def __init__(self, metadata: OperatorMetadata):
        """
        初始化算子
        
        Args:
            metadata: 算子元数据
        """
        self.metadata = metadata
    
    @abstractmethod
    def get_compute_complexity(self) -> float:
        """
        获取计算复杂度 (FLOPs)
        
        Returns:
            FLOPs数量
        """
        pass
    
    @abstractmethod
    def get_memory_requirement(self) -> Dict[str, int]:
        """
        获取内存需求
        
        Returns:
            内存需求字典 {
                'input': 输入内存大小(字节),
                'output': 输出内存大小(字节),
                'weight': 权重内存大小(字节)
            }
        """
        pass
    
    def get_io_volume(self) -> Dict[str, int]:
        """
        获取 I/O 数据量
        
        Returns:
            I/O 数据量字典 {
                'load': 加载数据量(字节),
                'store': 存储数据量(字节)
            }
        """
        io = self.metadata.io_config
        input_size = io.input_shape.size() * self.metadata.batch_size * io.input_dtype.value
        output_size = io.output_shape.size() * self.metadata.batch_size * io.output_dtype.value
        weight_size = io.weight_shape.size() * self.metadata.batch_size * io.weight_dtype.value
        
        return {
            'load': input_size + weight_size,
            'store': output_size,
        }


class MatmulOperator(BaseOperator):
    """矩阵乘法算子"""
    
    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)
    
    def get_compute_complexity(self) -> float:
        """计算矩阵乘法的 FLOPs"""
        io = self.metadata.io_config
        # FLOPs = 2 * m * n * k (因为矩阵乘法需要m*n*k次乘法和加法)
        m = io.input_shape.m
        k = io.input_shape.n
        n = io.output_shape.n
        batch = self.metadata.batch_size
        
        return 2.0 * m * k * n * batch
    
    def get_memory_requirement(self) -> Dict[str, int]:
        """获取矩阵乘法的内存需求"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size
        
        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        weight_mem = io.weight_shape.size() * io.weight_dtype.value
        
        return {
            'input': input_mem,
            'output': output_mem,
            'weight': weight_mem,
        }


class AttentionOperator(BaseOperator):
    """注意力算子基类"""
    
    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)
    
    def get_compute_complexity(self) -> float:
        """计算注意力的 FLOPs"""
        io = self.metadata.io_config
        # Q-K 矩阵乘法
        seq_len = io.input_shape.m
        head_dim = io.input_shape.n
        
        # Q*K^T: seq_len * head_dim * seq_len
        # Softmax(Q*K^T) * V: seq_len * seq_len * head_dim
        # 总计: 2 * seq_len^2 * head_dim
        # print(f'Use here Attention FLOPs: {2.0 * seq_len * seq_len * head_dim}')
        return 4.0 * seq_len * seq_len * head_dim
    
    def get_memory_requirement(self) -> Dict[str, int]:
        """获取注意力的内存需求"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size
        
        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        # 注意力中间结果（Q*K^T）
        intermediate_mem = io.input_shape.m * io.input_shape.m * batch * DataType.FP32.value
        
        return {
            'input': input_mem,
            'output': output_mem,
            'intermediate': intermediate_mem,
        }


class FFNOperator(BaseOperator):
    """前馈网络算子"""
    
    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)
    
    def get_compute_complexity(self) -> float:
        """计算 FFN 的 FLOPs"""
        io = self.metadata.io_config
        m = io.input_shape.m
        k = io.input_shape.n
        n = io.output_shape.n
        batch = self.metadata.batch_size
        
        # FFN: input * w1 + w1_out * w2
        # 通常 w1 是 gate+up projection，计算两次矩阵乘法
        return 2.0 * 2.0 * m * k * n * batch
    
    def get_memory_requirement(self) -> Dict[str, int]:
        """获取 FFN 的内存需求"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size
        
        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        weight_mem = io.weight_shape.size() * io.weight_dtype.value
        
        return {
            'input': input_mem,
            'output': output_mem,
            'weight': weight_mem,
        }


class TransferOperator(BaseOperator):
    """数据传输算子基类"""
    
    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)
    
    def get_compute_complexity(self) -> float:
        """传输算子没有计算复杂度"""
        return 0.0
    
    def get_memory_requirement(self) -> Dict[str, int]:
        """获取传输的数据量"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size
        
        transfer_size = io.input_shape.size() * batch * io.input_dtype.value
        
        return {
            'transfer': transfer_size,
        }


# 算子类型注册表
OPERATOR_REGISTRY: Dict[str, type] = {
    'matmul': MatmulOperator,
    'attention': AttentionOperator,
    'ffn': FFNOperator,
    'transfer': TransferOperator,
}


def create_operator(op_type: str, metadata: OperatorMetadata) -> BaseOperator:
    """
    工厂函数 - 创建算子实例
    
    Args:
        op_type: 算子类型
        metadata: 算子元数据
        
    Returns:
        算子实例
    """
    operator_class = OPERATOR_REGISTRY.get(op_type, MatmulOperator)
    return operator_class(metadata)
