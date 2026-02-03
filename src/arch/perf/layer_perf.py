from dataclasses import dataclass, field
from typing import List

from src.arch.perf.op_perf import OperatorPerformance


@dataclass
class LayerPerformance:
    """单层的性能指标"""

    layer_name: str = ""
    layer_type: str = ""

    # 算子列表
    operators: List[OperatorPerformance] = field(default_factory=list)

    # 聚合指标
    total_compute_time: float = 0.0
    total_memory_time: float = 0.0
    total_transfer_time: float = 0.0
    total_time: float = 0.0
    # 该层中每个算子的总时间
    one_op_total_time: dict[str, float] = field(default_factory=dict)


    def add_operator(self, op_perf: OperatorPerformance) -> None:
        """添加算子性能"""
        self.operators.append(op_perf)
        self.total_compute_time += op_perf.compute_time
        self.total_memory_time += op_perf.memory_time
        self.total_transfer_time += op_perf.transfer_time
        self.one_op_total_time[op_perf.name] = max(op_perf.compute_time, op_perf.memory_time) + op_perf.transfer_time

    def finalize(self) -> None:
        """计算最终的总时间"""
        # 总时间取计算和内存的最大值加上传输时间
        self.total_time = (
            max(self.total_compute_time, self.total_memory_time)
            + self.total_transfer_time
        )
