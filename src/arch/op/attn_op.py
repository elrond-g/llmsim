from typing import Dict

from hardware.hardware_config import HardwareConfig
from src.arch.op.operator_base import BaseOperator, DataType, OperatorMetadata


class AttentionOperator(BaseOperator):
    """注意力算子基类"""

    def __init__(self, metadata: OperatorMetadata):
        super().__init__(metadata)

    def get_compute_complexity(self) -> float:
        """计算注意力的 FLOPs"""

        # Q*K^T: seq_len * head_dim * seq_len
        # Softmax(Q*K^T) * V: seq_len * seq_len * head_dim
        # 总计: 2 * seq_len^2 * head_dim
        # print(f'Use here Attention FLOPs: {2.0 * seq_len * seq_len * head_dim}')
        # return 4.0 * seq_len * seq_len * head_dim

        def _legacy_cal_mac_time(
            cal_count: int, dtype: int, mac_int8: float = 500.0
        ) -> float:
            """
            old_main 风格的 MAC 时间计算
            返回: 微秒 (us)
            """
            return 2 * cal_count / mac_int8 / 1000000.0 * dtype

        operate_io = self.metadata.io_config
        m, n, k = (
            operate_io.input_shape.m,
            operate_io.input_shape.n,
            operate_io.output_shape.n,
        )
        _count = m * n * k * self.metadata.batch_size
        compute_time = _legacy_cal_mac_time(_count, operate_io.weight_dtype.value)
        return compute_time

    def get_memory_requirement(self) -> Dict[str, int]:
        """获取注意力的内存需求"""
        io = self.metadata.io_config
        batch = self.metadata.batch_size

        input_mem = io.input_shape.size() * batch * io.input_dtype.value
        output_mem = io.output_shape.size() * batch * io.output_dtype.value
        # 注意力中间结果（Q*K^T）
        intermediate_mem = (
            io.input_shape.m * io.input_shape.m * batch * DataType.FP32.value
        )

        return {
            "input": input_mem,
            "output": output_mem,
            "intermediate": intermediate_mem,
        }

    def get_hbm_time(self, hardware: HardwareConfig) -> float:
        pass
