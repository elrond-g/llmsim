from typing import Dict, Optional

from src.arch.model_type import AttentionType
from src.arch.op.attn_op import AttentionOperator
from src.arch.op.ffn_op import FFNOperator
from src.arch.op.matmul_op import MatmulOperator
from src.arch.op.mla_attn_op import MLAAttentionOperator
from src.arch.op.operator_base import BaseOperator, OperatorMetadata
from src.arch.op.transfer_op import TransferOperator

# Operator registry supporting multiple implementations - two-level structure
# Level 1: Operator type (attention, matmul, ffn, transfer)
# Level 2: Specific implementation variants (mha, mla, linear, etc.)
# Values can be None (indicating not implemented) or specific operator classes
OPERATOR_REGISTRY: Dict[str, Dict[str, Optional[type]]] = {
    "matmul": {
        "default": MatmulOperator,
    },
    "attention": {
        "mha": None,  # Multi-Head Attention - to be implemented
        "mla": MLAAttentionOperator,  # Multi-Head Latent Attention - to be implemented
        "linear": None,  # Linear Attention - to be implemented
        "hybrid": None,  # Hybrid Attention - to be implemented
        "default": AttentionOperator,  # Default to base class
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
    Get the operator class corresponding to the attention type

    Args:
        attention_type: Attention type enum

    Returns:
        Corresponding attention operator class, or default operator class if not found
    """
    attention_map = {
        AttentionType.MHA: "mha",
        AttentionType.MLA: "mla",
        AttentionType.LINEAR: "linear",
        AttentionType.HYBRID: "hybrid",
    }

    sub_key = attention_map.get(attention_type, "default")
    operator_class = OPERATOR_REGISTRY["attention"].get(sub_key)

    # Fallback to default implementation if specific implementation not available
    if operator_class is None:
        operator_class = OPERATOR_REGISTRY["attention"]["default"]

    return operator_class


def create_operator(
    op_type: str,
    metadata: OperatorMetadata,
    attention_type: Optional[AttentionType] = None,
) -> BaseOperator:
    """
    Factory function - Create operator instance

    Args:
        op_type: Operator type (matmul, attention, ffn, transfer)
        metadata: Operator metadata
        attention_type: Attention type (only valid for attention operators)

    Returns:
        Operator instance
    """
    if op_type not in OPERATOR_REGISTRY:
        # Unknown operator type, fallback to matmul
        op_type = "matmul"

    if op_type == "attention" and attention_type is not None:
        # Select specific implementation based on attention_type
        operator_class = _get_attention_operator_class(attention_type)
        assert operator_class is not None, "Attention operator class should not be None"
    else:
        # Use default implementation
        impl_dict = OPERATOR_REGISTRY[op_type]
        operator_class = impl_dict.get("default")
        assert (
            operator_class is not None
        ), f"No default operator found for type: {op_type}"

    return operator_class(metadata)
