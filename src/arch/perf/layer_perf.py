from dataclasses import dataclass, field
from typing import List

from src.arch.perf.op_perf import OperatorPerformance


@dataclass
class LayerPerformance:
    """Performance metrics for a single layer"""

    layer_name: str = ""
    layer_type: str = ""

    # Operator list
    operators: List[OperatorPerformance] = field(default_factory=list)

    # Aggregated metrics
    total_compute_time: float = 0.0
    total_memory_time: float = 0.0
    total_transfer_time: float = 0.0
    total_time: float = 0.0
    # Total time for each operator in this layer
    one_op_total_time: dict[str, float] = field(default_factory=dict)
    layer_total_mem_occupy: float = 0.0

    def add_operator(self, op_perf: OperatorPerformance) -> None:
        """Add operator performance"""
        self.operators.append(op_perf)
        self.total_compute_time += op_perf.compute_time
        self.total_memory_time += op_perf.memory_time
        self.total_transfer_time += op_perf.transfer_time
        self.one_op_total_time[op_perf.name] = (
            max(op_perf.compute_time, op_perf.memory_time) + op_perf.transfer_time
        )
        self.layer_total_mem_occupy += op_perf.weight_mem_occupy

    def finalize(self) -> None:
        """Calculate final total time"""
        # Total time takes the maximum of compute and memory plus transfer time
        self.total_time = (
            max(self.total_compute_time, self.total_memory_time)
            + self.total_transfer_time
        )
