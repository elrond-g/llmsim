from dataclasses import dataclass, field
from typing import List

from arch.config import ScheduleConfig
from src.arch.perf.layer_perf import LayerPerformance
from src.arch.model_type import ForwardMode

@dataclass
class ModelPerformance:
    """整个模型的性能指标"""

    model_name: str = ""
    forward_mode: str = ""

    # 层级性能
    layer_performances: List[LayerPerformance] = field(default_factory=list)

    # 聚合指标
    total_compute_time: float = 0.0
    total_memory_time: float = 0.0
    total_transfer_time: float = 0.0
    total_time: float = 0.0
    ttft: float = 0.0
    throughput: float = 0.0

    schedule_config: ScheduleConfig = field(default_factory=ScheduleConfig)

    def add_layer(self, layer_perf: LayerPerformance) -> None:
        """添加层性能"""
        self.layer_performances.append(layer_perf)

    def finalize(self) -> None:
        """计算最终指标"""
        sum_full_time = 0.0  # 所有算子的 full_time 之和（微秒），用于百分比计算

        for layer_perf in self.layer_performances:
            layer_perf.finalize()
            # 总时间取计算时间和内存时间的最大值（因为会流水线执行），加上传输时间
            layer_total = (
                max(layer_perf.total_compute_time, layer_perf.total_memory_time)
                + layer_perf.total_transfer_time
            )
            self.total_compute_time += layer_perf.total_compute_time
            self.total_memory_time += layer_perf.total_memory_time
            self.total_transfer_time += layer_perf.total_transfer_time
            # 总时间取所有层的最大值之和
            self.total_time += layer_total

        # 计算所有算子的 full_time 之和（用于百分比计算，与旧版本一致）
        # full_time = per_layer_time * layers，单位是微秒
        for layer_perf in self.layer_performances:
            for op_perf in layer_perf.operators:
                # 算子的 total_time 已经是 per_layer_time * layer_count 的结果（毫秒）
                # 需要乘以 1000 转换为微秒
                full_time_us = op_perf.total_time * 1000.0
                sum_full_time += full_time_us
                self.ttft += op_perf.total_time

        # 如果总时间为0，使用 sum_full_time
        if self.total_time == 0:
            self.total_time = sum_full_time / 1000.0  # 转换为毫秒

        # 保存 sum_full_time 用于百分比计算（微秒）
        self._sum_full_time = sum_full_time

    def get_bottleneck_op(self) -> tuple:
        """获取性能瓶颈算子"""
        max_time = 0.0
        bottleneck_op = None

        for layer_perf in self.layer_performances:
            for op_perf in layer_perf.operators:
                if op_perf.total_time > max_time:
                    max_time = op_perf.total_time
                    bottleneck_op = (layer_perf.layer_name, op_perf.name, op_perf)

        return bottleneck_op

    def get_percentage(self, time_us: float) -> float:
        """获取所占百分比（基于所有算子full_time之和）"""
        sum_full_time = getattr(self, "_sum_full_time", self.total_time)
        if sum_full_time == 0:
            return 0.0
        return time_us / sum_full_time * 100


    def get_ttft_or_tpot(self) -> float:
        """TTFT ms， 0.02 是为了考虑下框架层面的开销"""
        return self.ttft * 1.02

    def get_throughput(self) -> float:
        """Throughput TPS 吞吐量: tokens/second， 需要切分prefill 还是 decode"""
        """
        计算吞吐量 (tokens/second)
        Prefill 模式: (batch_size * seq_len) / TTFT
        Decode 模式: batch_size / time_per_token
        """
        mode = self.schedule_config.mode
        if mode == ForwardMode.EXTEND:  # Prefill
            total_tokens = self.schedule_config.batch_size * self.schedule_config.max_seqlen
            ttft_seconds = self.get_ttft_or_tpot() / 1000.0
            return total_tokens / ttft_seconds if ttft_seconds > 0 else 0.0
        else:  # Decode
            # Decode 阶段：每个 token 的生成时间
            # TODO 需要调整
            time_per_token_ms = self.get_ttft_or_tpot()   # 假设 total_time 是 per-token 时间
            time_per_token_s = time_per_token_ms / 1000.0
            return self.schedule_config.batch_size / time_per_token_s if time_per_token_s > 0 else 0.0

    def get_throughput_single_gpu(self) -> float:
        return self.get_throughput() / (self.schedule_config.tp_size * self.schedule_config.dp_size)
