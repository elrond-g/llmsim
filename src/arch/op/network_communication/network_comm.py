from src.arch.config import ModelConfig, ScheduleConfig
from src.arch.op.operator_base import DataType, Tensor
from src.hardware.hardware_config import HardwareConfig


class NetworkComm:

    def __init__(
        self,
        hardware_config: "HardwareConfig",
        model_config: "ModelConfig",
        schedule_config: "ScheduleConfig",
    ):
        self.hardware_config = hardware_config
        self.model_config = model_config
        self.schedule_config = schedule_config

    # calculate the bandwidth cost of a tensor in bytes. it will distinguish inter-node and intra-node
    # when inter_node is True, it will calculate the inter-node bandwidth cost using link_bandwidth, otherwise it will calculate the intra-node bandwidth cost
    def size_of_bandwidth(
        self,
        tensor: Tensor,
        dtype: DataType,
        rdma_bandwidth,
        link_bandwidth,
        inter_node: bool = False,
    ):
        # if it only have one gpu, then there is no bandwidth cost
        if self.schedule_config.world_size <= 1:
            return 0
        size = 1
        if dtype == DataType.FP32:
            size = 4
        elif dtype == DataType.FP16 or dtype == DataType.BF16:
            size = 2
        elif dtype == DataType.INT8 or dtype == DataType.FP8:
            size = 1
        size *= tensor.size()
        if inter_node:
            return size / (1024**3) / rdma_bandwidth

        return size / (1024**3) / link_bandwidth
