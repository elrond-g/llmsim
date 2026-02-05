from op.operator_base import DataType

from arch.config import ModelConfig


def mha_gqa_kvcache(config: ModelConfig, kvcache_dtype: DataType):
    """
    KVCache size for all layers
    """
    decoder_layers = getattr(config, "num_hidden_layers")
    assert decoder_layers > 0, "decoder_layers must be greater than 0"
    number_of_kv_heads = getattr(config, "num_key_value_heads")
    assert number_of_kv_heads > 0, "number_of_kv_heads must be greater than 0"
    assert number_of_kv_heads <= getattr(
        config, "num_attention_heads"
    ), "number_of_kv_heads must be less than or equal to num_attention_heads"
    head_dim = getattr(config, "hidden_size")
    assert head_dim > 0, "head_dim must be greater than 0"
    kv_cache_size = 2 * decoder_layers * number_of_kv_heads * head_dim

    kv_cache_size = kv_cache_size * kvcache_dtype.value

    return kv_cache_size


def mha_gqa_kvcache_per_gpu(config: ModelConfig, kvcache_dtype: DataType, tp_size: int):
    """
    Calculate KVCache size for a single GPU based on TP group partitioning
    """
    return mha_gqa_kvcache(config, kvcache_dtype) // tp_size


def mla_kvcache(config: ModelConfig, kvcache_dtype: DataType):
    """
    KVCache size for all layers
    """
    decoder_layers = getattr(config, "num_hidden_layers")
    assert decoder_layers > 0, "decoder_layers must be greater than 0"
    kv_lora_rank = getattr(config, "kv_lora_rank")
    assert kv_lora_rank > 0, "kv_lora_rank must be greater than 0"
    qk_rope_head_dim = getattr(config, "qk_rope_head_dim")
    assert qk_rope_head_dim > 0, "qk_rope_head_dim must be greater than 0"

    kv_cache_size = decoder_layers * (kv_lora_rank + qk_rope_head_dim)
    kv_cache_size = kv_cache_size * kvcache_dtype.value
    return kv_cache_size


def mla_kvcache_per_gpu(config: ModelConfig, kvcache_dtype: DataType, tp_size: int):
    """
    Calculate MLA KVCache size for a single GPU based on TP group partitioning

    MLA KVCache partitioning strategy under TP parallelism:
    - MLA uses low-rank compressed representation (kv_lora_rank + qk_rope_head_dim), no longer in multi-head form
    - Only Query heads are partitioned by TP, KVCache itself is not partitioned, all TP ranks store the same compressed KV Cache
    - Therefore, the KVCache size for a single GPU is the same as the total size

    Note: If the implementation strategy changes in the future, this function needs to be adjusted accordingly
    """
    return mla_kvcache(config, kvcache_dtype)
