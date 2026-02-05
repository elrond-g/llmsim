"""
Performance calculation engine - Unified performance computation and analysis module
"""

from src.arch.models_arch.model_arch import BaseModelArch
from src.arch.op.operator_base import BaseOperator
from src.arch.perf.layer_perf import LayerPerformance
from src.arch.perf.model_perf import ModelPerformance
from src.arch.perf.op_perf import OperatorPerformance
from src.hardware.hardware_config import HardwareConfig


class PerformanceCalculator:
    """Performance calculation engine"""

    def __init__(self, hardware_config: HardwareConfig):
        """
        Initialize performance calculator

        Args:
            hardware_config: Hardware configuration
        """
        self.hardware = hardware_config

    def calculate_compute_time(self, operator: BaseOperator) -> float:
        """
        Calculate operator computation time

        Args:
            operator: Operator instance

        Returns:
            Computation time (microseconds)
        """
        flops = operator.get_compute_complexity()

        if flops == 0:
            return 0.0

        # Select appropriate MAC performance based on data type
        io_config = operator.metadata.io_config
        dtype_bytes = io_config.weight_dtype.value

        if dtype_bytes == 1:  # INT8
            mac_gflops = self.hardware.compute.mac_int8_gflops
        elif dtype_bytes == 4:  # FP32
            mac_gflops = self.hardware.compute.mac_fp32_gflops
        else:  # BF16/FP16 (默认)
            mac_gflops = self.hardware.compute.mac_bf16_gflops

        # 时间 = FLOPs / (GFLOPS * 10^9) * 10^6 = FLOPs / (GFLOPS * 1000 * 1000)
        compute_time_us = flops / (mac_gflops * 1e6)

        return compute_time_us

    def calculate_attention_hbm_time(
        self,
        load_count: int,
        load_dtype: int,
        store_count: int,
        store_dtype: int,
        dma: float,
    ) -> float:
        """
        Returns: microseconds (us)
        """
        return (load_count * load_dtype + store_count * store_dtype) / dma / 1000000.0

    def calculate_memory_time(self, operator: BaseOperator) -> float:
        """
        Calculate operator memory access time

        Args:
            operator: Operator instance

        Returns:
            Memory time (microseconds)
        """
        io_volume = operator.get_io_volume()

        load_bytes = io_volume.get("load", 0)
        store_bytes = io_volume.get("store", 0)
        memory_time_us = (
            (load_bytes + store_bytes)
            / self.hardware.bandwidth.hbm_bandwidth_gb_s
            / 1e6
        )
        return memory_time_us

    def calculate_transfer_time(self, operator: BaseOperator) -> float:
        """
        Calculate transfer operator transmission time

        Args:
            operator: Operator instance
            bandwidth_gb_s: Bandwidth (GB/s), if None use default network bandwidth

        Returns:
            Transfer time (microseconds)
        """
        # Get bandwidth (from operator's custom attribute or use default)
        bandwidth_gb_s = getattr(operator, "_bandwidth_gb_s", None)
        if bandwidth_gb_s is None:
            # Select appropriate bandwidth based on operator name
            if operator.metadata.name == "dispatch":
                # dispatch uses nb config from DeepSeek legacy version (18.58 in decode mode)
                bandwidth_gb_s = self.hardware.bandwidth.dma_bandwidth_decode_gb_s
            elif operator.metadata.name == "combine":
                # combine uses nb config from DeepSeek legacy version (22.64 in decode mode)
                bandwidth_gb_s = self.hardware.bandwidth.dma_bandwidth_decode_gb_s
            else:
                bandwidth_gb_s = self.hardware.bandwidth.link_bandwidth_gb_s

        io_volume = operator.get_io_volume()
        transfer_bytes = io_volume.get("transfer", io_volume.get("load", 0))

        # data.transfer = m * n * batch * dtype / nb / 1000.0
        # Here transfer_bytes = m * n * batch * dtype (bytes), nb = GB/s
        # Returns "microseconds/layer"
        transfer_time_us = transfer_bytes / bandwidth_gb_s / 1000.0

        return transfer_time_us

    def calculate_operator_performance(
        self, operator: BaseOperator
    ) -> OperatorPerformance:
        """
        Calculate performance metrics for a single operator

        Args:
            operator: Operator instance

        Returns:
            Operator performance metrics
        """
        metadata = operator.metadata

        # Calculate different types of time
        compute_time = 0.0
        memory_time = 0.0
        transfer_time = 0.0

        if metadata.op_type == "transfer":
            transfer_time = self.calculate_transfer_time(operator)
        elif metadata.op_type == "attention":
            compute_time = operator.get_compute_complexity()
            memory_time = operator.get_hbm_time(hardware=self.hardware)
        elif metadata.op_type == "matmul":
            compute_time = self.calculate_compute_time(operator)
            memory_time = self.calculate_memory_time(operator)
            # print(f'name = {operator.metadata.name}, compute_time = {compute_time}, memory_time={memory_time}')
        else:
            # Unknown operator type, log warning but continue execution
            print(
                f"Warning: Unrecognized operator type '{metadata.op_type}', operator name: {metadata.name}"
            )

        # Time per layer (multiplied by layer count)
        layer_count = metadata.num_layers

        # Add null check to prevent NoneType errors
        if compute_time is None:
            print(
                f"Warning: compute_time is None for operator {metadata.name}, setting to 0.0"
            )
            compute_time = 0.0
        if memory_time is None:
            print(
                f"Warning: memory_time is None for operator {metadata.name}, setting to 0.0"
            )
            memory_time = 0.0
        if transfer_time is None:
            print(
                f"Warning: transfer_time is None for operator {metadata.name}, setting to 0.0"
            )
            transfer_time = 0.0

        total_compute_time = compute_time * layer_count
        total_memory_time = memory_time * layer_count
        total_transfer_time = transfer_time * layer_count

        # Total time takes the maximum value
        total_time = max(total_compute_time, total_memory_time) + total_transfer_time

        single_layer_op_weight_mem = operator.get_weight_mem_occupy()
        op_weight_mem = single_layer_op_weight_mem * layer_count

        op_perf = OperatorPerformance(
            name=metadata.name,
            op_type=metadata.op_type,
            compute_time=compute_time,
            memory_time=memory_time,
            transfer_time=transfer_time,
            op_time_single_layer=max(compute_time, memory_time) + transfer_time,
            total_time=total_time / 1000.0,  # Convert to milliseconds
            flops=operator.get_compute_complexity() * layer_count,
            memory_volume=operator.get_memory_requirement().get("weight", 0),
            io_volume=operator.get_io_volume().get("load", 0)
            + operator.get_io_volume().get("store", 0),
            metadata=metadata,
            weight_mem_occupy=op_weight_mem,
        )

        return op_perf

    def calculate_model_performance(
        self, model_arch: BaseModelArch
    ) -> ModelPerformance:
        """
        Calculate performance for the entire model

        Args:
            model_arch: Model architecture instance

        Returns:
            Model performance metrics
        """
        model_arch.build_operators()

        model_perf = ModelPerformance(
            model_name=model_arch.model_config.model_type,
            forward_mode=model_arch.schedule_config.mode.name,
            schedule_config=model_arch.schedule_config,
        )

        # Most matrix operators are here, attention operators in attention section, transfer operators in transfer section
        for operator in model_arch.operators:
            op_perf = self.calculate_operator_performance(operator)
            layer_perf = LayerPerformance(layer_name=op_perf.name, layer_type="compute")
            layer_perf.add_operator(op_perf)
            model_perf.add_layer(layer_perf)

        # Process attention operators
        for attn_key, operators in model_arch.attention_operators.items():
            for operator in operators:
                op_perf = self.calculate_operator_performance(operator)
                layer_perf = LayerPerformance(
                    layer_name=attn_key, layer_type="attention"
                )
                layer_perf.add_operator(op_perf)
                model_perf.add_layer(layer_perf)

        # Process transfer operators
        for operator in model_arch.transfer_operators:
            op_perf = self.calculate_operator_performance(operator)
            layer_perf = LayerPerformance(
                layer_name=op_perf.name, layer_type="transfer"
            )
            layer_perf.add_operator(op_perf)
            model_perf.add_layer(layer_perf)

        model_perf.finalize()

        return model_perf

    def print_performance_report(
        self,
        model_perf: ModelPerformance,
        output_format: str = "console",
        output_path: str = None,
    ) -> None:
        """
        Print performance report

        Args:
            model_perf: Model performance metrics
            output_format: Output format ('console' or 'excel')
            output_path: Output file path (optional, only valid for certain formats)
        """
        from src.visual.report_formatter import create_formatter

        formatter = create_formatter(output_format)

        if output_path:
            formatter.save(model_perf, output_path)
        else:
            formatter.save(model_perf)
