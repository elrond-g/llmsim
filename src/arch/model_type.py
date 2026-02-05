from enum import Enum


class AttentionType(Enum):
    """Attention type"""

    MHA = "mha"  # Multi-Head Attention
    MLA = "mla"  # Multi-Head Latent Attention
    HYBRID = "hybrid"  # Hybrid Attention
    LINEAR = "linear"  # Linear Attention


class ForwardMode(Enum):
    """Forward pass mode"""

    EXTEND = 0  # Sequence extension mode
    DECODE = 1  # Decoding mode
