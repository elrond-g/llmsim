from abc import ABC, abstractmethod
from typing import Dict, List

from src.arch.config import ForwardMode, ModelConfig, ScheduleConfig
from src.arch.op.operator_base import BaseOperator


class BaseModelArch(ABC):
    """Base class for model architecture"""

    def __init__(self, model_config: ModelConfig, schedule_config: ScheduleConfig):
        """
        Initialize model architecture. A model structure requires at least model config, schedule config, operators, attention operators, and transfer operators.

        Args:
            model_config: Model configuration
            schedule_config: Schedule configuration
        """
        self.model_config = model_config
        self.schedule_config = schedule_config
        self.operators: List[BaseOperator] = []
        self.attention_operators: Dict[str, List[BaseOperator]] = {}
        self.transfer_operators: List[BaseOperator] = []

    @abstractmethod
    def build_operators(self) -> List[BaseOperator]:
        """Build the operator graph for the model"""
        pass

    def get_seq_length(self) -> int:
        """Get sequence length based on mode"""
        if self.schedule_config.mode == ForwardMode.EXTEND:
            return self.schedule_config.max_seqlen
        elif self.schedule_config.mode == ForwardMode.DECODE:
            return self.schedule_config.batch_size
        return self.schedule_config.max_seqlen

    def _add_operator(self, operator: BaseOperator) -> None:
        """Add operator to the operator list"""
        self.operators.append(operator)

    def _add_attention_operator(self, key: str, operators: List[BaseOperator]) -> None:
        """Add attention operator"""
        self.attention_operators[key] = operators

    def _add_transfer_operator(self, operator: BaseOperator) -> None:
        """Add transfer operator"""
        self.transfer_operators.append(operator)

    def get_kv_cache(self):
        """KVCache theoretically only depends on model config"""
        pass

    def get_kv_cache_per_gpu(self):
        """KVCache theoretically only depends on model config"""
        pass
