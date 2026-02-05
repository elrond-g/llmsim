"""
Hardware configuration module - Supports dynamic loading of hardware parameters from JSON/JSON5 configuration files
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DeviceType(Enum):
    """Device type"""

    UNKNOWN = "unknown"
    GPU = "gpu"
    XPU = "xpu"
    ACCELERATOR = "accelerator"


@dataclass
class MemoryConfig:
    """Memory configuration"""

    hbm_size_gb: int = 96  # HBM memory size (GB)
    cache_line_size: int = 128  # Cache line size


@dataclass
class BandwidthConfig:
    """Bandwidth configuration"""

    hbm_bandwidth_gb_s: float = 1.8  # HBM bandwidth (TB/s)
    dma_bandwidth_gb_s: float = (
        85.0  # DMA bandwidth (GB/s) - extend mode, bandwidth used by DISPATCH/COMBINE
    )
    dma_bandwidth_decode_gb_s: float = 22.64  # DMA bandwidth (GB/s) - decode mode

    # Intra-node communication (multiple cards within same machine, via Link such as NVLink)
    link_bandwidth_gb_s: float = 85.0  # Link bandwidth (GB/s) - extend mode
    link_bandwidth_decode_gb_s: float = 22.64  # Link bandwidth (GB/s) - decode mode

    # Inter-node communication (different machines, via RDMA such as InfiniBand)
    rdma_bandwidth_gb_s: float = 85.0  # RDMA bandwidth (GB/s) - extend mode
    rdma_bandwidth_decode_gb_s: float = 22.64  # RDMA bandwidth (GB/s) - decode mode

    @property
    def network_bandwidth_decode_gb_s(self) -> float:
        """Backward compatibility: network bandwidth defaults to Link bandwidth"""
        return self.link_bandwidth_decode_gb_s


@dataclass
class ComputeConfig:
    """Compute configuration"""

    mac_int8_gflops: float = 500.0  # INT8 MAC performance (TFLOPS)
    mac_fp32_gflops: float = 125.0  # FP32 MAC performance (TFLOPS)
    mac_bf16_gflops: float = 250.0  # BF16 MAC performance (TFLOPS)


@dataclass
class HardwareConfig:
    """Hardware configuration container"""

    device_type: DeviceType = DeviceType.GPU
    name: str = "Default GPU"

    memory: Optional[MemoryConfig] = None
    bandwidth: Optional[BandwidthConfig] = None
    compute: Optional[ComputeConfig] = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.bandwidth is None:
            self.bandwidth = BandwidthConfig()
        if self.compute is None:
            self.compute = ComputeConfig()

    @staticmethod
    def _parse_bandwidth_config(bandwidth_data: dict) -> "BandwidthConfig":
        """Parse bandwidth configuration, supporting both old and new formats"""

        # If link/rdma config exists in new format, use them; otherwise fallback to network_bandwidth
        return BandwidthConfig(
            hbm_bandwidth_gb_s=bandwidth_data.get("hbm_bandwidth_gb_s", 1.8),
            dma_bandwidth_gb_s=bandwidth_data.get("dma_bandwidth_gb_s", 85.0),
            dma_bandwidth_decode_gb_s=bandwidth_data.get(
                "dma_bandwidth_decode_gb_s", 22.64
            ),
            link_bandwidth_gb_s=bandwidth_data.get("link_bandwidth_gb_s", 85),
            link_bandwidth_decode_gb_s=bandwidth_data.get(
                "link_bandwidth_decode_gb_s", 22.64
            ),
            rdma_bandwidth_gb_s=bandwidth_data.get("rdma_bandwidth_gb_s", 85),
            rdma_bandwidth_decode_gb_s=bandwidth_data.get(
                "rdma_bandwidth_decode_gb_s", 22.64
            ),
        )

    @classmethod
    def from_json(cls, config_path: str) -> "HardwareConfig":
        """
        Load hardware configuration from JSON/JSON5 configuration file

        Args:
            config_path: JSON or JSON5 configuration file path

        Returns:
            HardwareConfig instance
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Hardware config not found: {config_path}")

        # 根据文件扩展名选择解析器
        if config_path.endswith(".json5"):
            import json5

            with open(config_path, "r") as f:
                data = json5.load(f)
        else:
            with open(config_path, "r") as f:
                data = json.load(f)

        # 解析嵌套配置
        memory_data = data.get("memory", {})
        bandwidth_data = data.get("bandwidth", {})
        compute_data = data.get("compute", {})

        return cls(
            device_type=DeviceType(data.get("device_type", "gpu")),
            name=data.get("name", "Unknown Hardware"),
            memory=MemoryConfig(
                hbm_size_gb=memory_data.get("hbm_size_gb", 96),
                cache_line_size=memory_data.get("cache_line_size", 128),
            ),
            bandwidth=cls._parse_bandwidth_config(bandwidth_data),
            compute=ComputeConfig(
                mac_int8_gflops=compute_data.get("mac_int8_gflops", 500.0),
                mac_fp32_gflops=compute_data.get("mac_fp32_gflops", 125.0),
                mac_bf16_gflops=compute_data.get("mac_bf16_gflops", 250.0),
            ),
        )


# Hardware configuration registry - supports loading predefined configurations by name
_HARDWARE_REGISTRY: dict[str, str] = {
    "default": "hardware_config/default_gpu.json5",
    "h20": "hardware_config/h20.json5",
    "h800": "hardware_config/h800.json5",
    "gb200": "hardware_config/gb200.json5",
    "klx_p800": "hardware_config/klx_p800.json5",
}


def get_hardware_config(name: str = "default") -> HardwareConfig:
    """
    Get predefined hardware configuration by name

    Args:
        name: Hardware configuration name (default, h20, h800, gb200)

    Returns:
        HardwareConfig instance
    """
    config_path = _HARDWARE_REGISTRY.get(name.lower())
    if config_path is None:
        raise ValueError(
            f"Unknown hardware config: {name}. Available: {list(_HARDWARE_REGISTRY.keys())}"
        )

    # If relative path, add project root directory
    if not os.path.isabs(config_path):
        # Get parent directory of current file (i.e., project root directory)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(project_root, config_path)

    return HardwareConfig.from_json(config_path)


# Default hardware configuration (fallback configuration)
DEFAULT_HARDWARE = get_hardware_config("default")
