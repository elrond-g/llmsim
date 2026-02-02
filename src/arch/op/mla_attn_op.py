from typing import Dict

from hardware.hardware_config import HardwareConfig
from src.arch.op.operator_base import BaseOperator, DataType, OperatorMetadata


class MLAAttentionOperator(BaseOperator):
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
        intermediate_mem = (
            io.input_shape.m * io.input_shape.m * batch * DataType.FP32.value
        )

        return {
            "input": input_mem,
            "output": output_mem,
            "intermediate": intermediate_mem,
        }

    def _calculate_attention_hbm_time(
        self,
        load_count: int,
        load_dtype: int,
        store_count: int,
        store_dtype: int,
        dma: float,
    ) -> float:
        """
        返回: 微秒 (us)
        """
        return (load_count * load_dtype + store_count * store_dtype) / dma / 1000000.0

    def get_hbm_time(self, hardware: HardwareConfig) -> float:
        """获取注意力的 HBM 时间"""
        op_name = self.metadata.name
        if op_name == "qkv":
            load_count = (
                self.metadata.io_config.weight_shape.size() * self.metadata.batch_size
            )  # 右边矩阵情况
            store_count = (
                self.metadata.io_config.input_shape.m
                * self.metadata.io_config.weight_shape.n
                * self.metadata.batch_size
            )  # 左边矩阵情况
        else:
            load_count = (
                self.metadata.io_config.input_shape.size()
                + self.metadata.io_config.weight_shape.size()
            ) * self.metadata.batch_size
            store_count = 0
        memory_time = self._calculate_attention_hbm_time(
            load_count,
            self.metadata.io_config.input_dtype.value,
            store_count,
            self.metadata.io_config.output_dtype.value,
            hardware.bandwidth.hbm_bandwidth_gb_s,
        )
        print(f"load_count={load_count}, store_count={store_count}, hbm={memory_time}")
        return memory_time
