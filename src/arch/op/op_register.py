from typing import Dict, Optional

from src.arch.model_type import AttentionType
from src.arch.op.attn_op import AttentionOperator
from src.arch.op.ffn_op import FFNOperator
from src.arch.op.matmul_op import MatmulOperator
from src.arch.op.mla_attn_op import MLAAttentionOperator
from src.arch.op.operator_base import BaseOperator, OperatorMetadata
from src.arch.op.transfer_op import TransferOperator

# 支持多实现的算子注册表 - 两层结构
# 第一层: 算子类型 (attention, matmul, ffn, transfer)
# 第二层: 具体实现变体 (mha, mla, linear 等)
# 值可以是 None（表示待实现）或具体的算子类
OPERATOR_REGISTRY: Dict[str, Dict[str, Optional[type]]] = {
    "matmul": {
        "default": MatmulOperator,
    },
    "attention": {
        "mha": None,  # Multi-Head Attention - 待实现
        "mla": MLAAttentionOperator,  # Multi-Head Latent Attention - 待实现
        "linear": None,  # Linear Attention - 待实现
        "hybrid": None,  # Hybrid Attention - 待实现
        "default": AttentionOperator,  # 默认使用基类
    },
    "ffn": {
        "default": FFNOperator,
    },
    "transfer": {
        "default": TransferOperator,
    },
}


def _get_attention_operator_class(attention_type: AttentionType) -> Optional[type]:
    """
    根据注意力类型获取对应的算子类

    Args:
        attention_type: 注意力类型枚举

    Returns:
        对应的注意力算子类，如果未找到则返回默认算子类
    """
    attention_map = {
        AttentionType.MHA: "mha",
        AttentionType.MLA: "mla",
        AttentionType.LINEAR: "linear",
        AttentionType.HYBRID: "hybrid",
    }

    sub_key = attention_map.get(attention_type, "default")
    operator_class = OPERATOR_REGISTRY["attention"].get(sub_key)

    # 如果特定实现未实现，回退到默认实现
    if operator_class is None:
        operator_class = OPERATOR_REGISTRY["attention"]["default"]

    return operator_class


def create_operator(
    op_type: str,
    metadata: OperatorMetadata,
    attention_type: Optional[AttentionType] = None,
) -> BaseOperator:
    """
    工厂函数 - 创建算子实例

    Args:
        op_type: 算子类型 (matmul, attention, ffn, transfer)
        metadata: 算子元数据
        attention_type: 注意力类型（仅对 attention 算子有效）

    Returns:
        算子实例
    """
    if op_type not in OPERATOR_REGISTRY:
        # 未知的算子类型，回退到 matmul
        op_type = "matmul"

    if op_type == "attention" and attention_type is not None:
        # 根据 attention_type 选择具体实现
        operator_class = _get_attention_operator_class(attention_type)
        assert operator_class is not None, "Attention operator class should not be None"
    else:
        # 使用默认实现
        impl_dict = OPERATOR_REGISTRY[op_type]
        operator_class = impl_dict.get("default")
        assert (
            operator_class is not None
        ), f"No default operator found for type: {op_type}"

    return operator_class(metadata)
