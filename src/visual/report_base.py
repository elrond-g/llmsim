from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.arch.perf_calculator import ModelPerformance


class ReportFormatter(ABC):
    """Abstract base class for performance report formatters"""

    @abstractmethod
    def format(self, model_perf: ModelPerformance) -> Any:
        """
        Format performance report

        Args:
            model_perf: Model performance metrics

        Returns:
            Formatted output (content depends on specific implementation)
        """
        pass

    @abstractmethod
    def save(self, model_perf: ModelPerformance, output_path: str = None) -> None:
        """
        Save performance report to file or output

        Args:
            model_perf: Model performance metrics
            output_path: Output file path (optional)
        """
        pass

    def _collect_data(self, model_perf: ModelPerformance) -> List[Dict[str, Any]]:
        """
        Collect all performance data for formatter use

        Args:
            model_perf: Model performance metrics

        Returns:
            List containing all row data
        """
        all_rows = []
        for layer_perf in model_perf.layer_performances:
            for op_perf in layer_perf.operators:
                op_meta = op_perf.metadata
                time_us = op_perf.total_time * 1000.0
                percentage = model_perf.get_percentage(time_us)
                all_rows.append(
                    {
                        "name": op_perf.name,
                        "type": op_perf.op_type,
                        "m": op_meta.io_config.input_shape.m,
                        #'n': op_meta.io_config.output_shape.n,
                        "n": (
                            op_meta.io_config.weight_shape.n
                            if op_meta.io_config.weight_shape.n is not None
                            else 0
                        ),
                        "k": op_meta.io_config.input_shape.n,
                        "batch": op_meta.batch_size,
                        "layers": op_meta.num_layers,
                        "in_dtype": op_meta.io_config.input_dtype.name,
                        "out_dtype": op_meta.io_config.output_dtype.name,
                        "weight_dtype": op_meta.io_config.weight_dtype.name,
                        "compute": op_perf.compute_time,
                        "memory": op_perf.memory_time,
                        "transfer": op_perf.transfer_time,
                        "op_time_single_layer": op_perf.op_time_single_layer,
                        "total": op_perf.total_time,
                        "percent": percentage,
                        "op_weight_mem": op_perf.weight_mem_occupy,
                    }
                )
        return all_rows
