"""
硬件配置模块 - 定义硬件参数，避免硬编码
"""
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """设备类型"""
    UNKNOWN = "unknown"
    GPU = "gpu"
    ACCELERATOR = "accelerator"


@dataclass
class MemoryConfig:
    """显存配置"""
    hbm_size_gb: int = 96  # HBM显存大小(GB)
    cache_line_size: int = 128  # Cache行大小


@dataclass
class BandwidthConfig:
    """带宽配置"""
    hbm_bandwidth_gb_s: float = 1.8  # HBM带宽(T/s)
    dma_bandwidth_gb_s: float = 85.0  # DMA带宽(GB/s) - 扩展模式
    dma_bandwidth_decode_gb_s: float = 22.64  # DMA带宽(GB/s) - 解码模式
    network_bandwidth_gb_s: float = 85.0  # 网络带宽(GB/s)
    network_bandwidth_decode_gb_s: float = 22.64  # 网络带宽(GB/s) - 解码模式


@dataclass
class ComputeConfig:
    """计算配置"""
    mac_int8_gflops: float = 500.0  # INT8 MAC性能(GFLOPS)
    mac_fp32_gflops: float = 125.0  # FP32 MAC性能(GFLOPS)
    mac_bf16_gflops: float = 250.0  # BF16 MAC性能(GFLOPS)


@dataclass
class HardwareConfig:
    """硬件配置容器"""
    device_type: DeviceType = DeviceType.GPU
    name: str = "Default GPU"
    
    memory: MemoryConfig = None
    bandwidth: BandwidthConfig = None
    compute: ComputeConfig = None
    
    def __post_init__(self):
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.bandwidth is None:
            self.bandwidth = BandwidthConfig()
        if self.compute is None:
            self.compute = ComputeConfig()
    
    @classmethod
    def create_default_gpu(cls) -> "HardwareConfig":
        """创建默认 GPU 配置"""
        return cls(
            device_type=DeviceType.GPU,
            name="Default GPU",
            memory=MemoryConfig(),
            bandwidth=BandwidthConfig(),
            compute=ComputeConfig(),
        )
    
    @classmethod
    def create_special_gpu(cls, memory: MemoryConfig, bandwidth: BandwidthConfig, compute: ComputeConfig) -> "HardwareConfig":
        """创建特殊 GPU 配置"""
        return cls(
            device_type=DeviceType.GPU,
            name="GPU",
            memory=memory,
            bandwidth=bandwidth,
            compute=compute,
        )
    
    @classmethod
    def create_high_performance_gpu(cls) -> "HardwareConfig":
        """创建高性能 GPU 配置"""
        return cls(
            device_type=DeviceType.GPU,
            name="High Performance GPU",
            memory=MemoryConfig(hbm_size_gb=160),
            bandwidth=BandwidthConfig(
                hbm_bandwidth_gb_s=2.4,
                dma_bandwidth_gb_s=120.0,
                dma_bandwidth_decode_gb_s=30.0,
            ),
            compute=ComputeConfig(
                mac_int8_gflops=750.0,
                mac_fp32_gflops=187.5,
                mac_bf16_gflops=375.0,
            ),
        )


# 预定义的硬件配置
DEFAULT_HARDWARE = HardwareConfig.create_default_gpu()
