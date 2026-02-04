from op.operator_base import DataType

from arch.config import ModelConfig


def mha_gqa_kvcache(config: ModelConfig, kvcache_dtype: DataType):
    """
    所有层KVCache的大小
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
    按照TP组切分来计算单个GPU的KVCache大小
    """
    return mha_gqa_kvcache(config, kvcache_dtype) // tp_size


def mla_kvcache(config: ModelConfig, kvcache_dtype: DataType):
    """
    所有层KVCache的大小
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
    按照TP组切分来计算单个GPU的MLA KVCache大小

    MLA的KVCache在TP并行时的切分策略：
    - MLA使用低秩压缩表示(kv_lora_rank + qk_rope_head_dim)，不再是多头形式
    - 只有Query的头按TP切分，KVCache本身不切分，所有TP rank保存相同的压缩KV Cache
    - 因此单个GPU的KVCache大小与总大小相同

    注意：如果未来实现策略改变，需要相应调整此函数
    """
    return mla_kvcache(config, kvcache_dtype)
