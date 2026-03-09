"""
Model architecture abstraction layer - Defines generic model architecture interfaces and base classes
"""

from src.arch.config import ModelConfig, ScheduleConfig
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.models_arch.deepseek_v3_model_arch import DeepSeekV3Arch
from src.arch.models_arch.glm_moe_dsa_model_arch import GlmMoeDsaArch
from src.arch.models_arch.minimax_m2_model_arch import MiniMaxM2Arch
from src.arch.models_arch.qwen3_moe_model_arch import Qwen3MoEArch
from src.arch.models_arch.qwen3_5_moe_model_arch import Qwen3_5MoEArch
from src.arch.models_arch.simple_model_arch import SimpleTransformerArch


_MODEL_ARCH_MAP: dict[str, type[BaseModelArch]] = {
    "deepseek_v3": DeepSeekV3Arch,
    "deepseek_r1": DeepSeekV3Arch,
    "glm_moe_dsa": GlmMoeDsaArch,
    "minimax_m2": MiniMaxM2Arch,
    "qwen3": SimpleTransformerArch,
    "qwen3_moe": Qwen3MoEArch,
    "qwen3_5_moe": Qwen3_5MoEArch,
}


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
    arch_cls = _MODEL_ARCH_MAP.get(model_type, SimpleTransformerArch)
    return arch_cls(model_config, schedule_config)
