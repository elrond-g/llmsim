"""
Model architecture abstraction layer - Defines generic model architecture interfaces and base classes
"""

from src.arch.config import ModelConfig, ScheduleConfig
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.models_arch.deepseek_v3_model_arch import DeepSeekV3Arch
from src.arch.models_arch.qwen3_moe_model_arch import Qwen3MoEArch
from src.arch.models_arch.simple_model_arch import SimpleTransformerArch


def create_model_arch(
    model_config: ModelConfig, schedule_config: ScheduleConfig
) -> BaseModelArch:
    """
    Factory function - Create appropriate model architecture

    Args:
        model_config: Model configuration
        schedule_config: Schedule configuration

    Returns:
        Model architecture instance
    """
    model_type = model_config.model_type.lower()

    if model_type in ("deepseek_v3", "deepseek_r1"):
        return DeepSeekV3Arch(model_config, schedule_config)
    elif model_type == "qwen3_moe":
        return Qwen3MoEArch(model_config, schedule_config)
    elif model_type == "qwen3":
        return SimpleTransformerArch(model_config, schedule_config)
    else:
        # Default to standard Transformer architecture
        return SimpleTransformerArch(model_config, schedule_config)
