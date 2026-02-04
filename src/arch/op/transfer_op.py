from typing import Dict

from hardware.hardware_config import HardwareConfig
from src.arch.op.operator_base import BaseOperator, OperatorMetadata


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
            "transfer": transfer_size,
        }

    def get_hbm_time(self, hardware: HardwareConfig) -> float:
        return 0.0

    def get_weight_mem_occupy(self) -> float:
        return 0.0
