# KV Cache 计算逻辑

本文档详细说明 MHA、GQA、MLA 三种注意力机制的 KV Cache 计算逻辑及 TP 并行切分策略。

## 目录

- [MHA (Multi-Head Attention)](#mha-multi-head-attention)
- [GQA (Grouped Query Attention)](#gqa-grouped-query-attention)
- [MLA (Multi-head Latent Attention)](#mla-multi-head-latent-attention)

---

## MHA (Multi-Head Attention)

### 计算逻辑

MHA 中每个 Query 头都有独立的 Key 和 Value 头。

```python
kv_cache_size = 2 × decoder_layers × num_attention_heads × head_dim
```

其中：
- `2`: 分别存储 Key 和 Value
- `decoder_layers`: 解码器层数
- `num_attention_heads`: 注意力头数
- `head_dim`: 每个头的维度 (hidden_size)

### TP 并行切分

MHA 的 KV Cache 按头数切分：

```python
kv_cache_per_gpu = kv_cache_size // tp_size
```

每个 GPU 存储 `num_attention_heads // tp_size` 个 KV 头。

---

## GQA (Grouped Query Attention)

### 计算逻辑

GQA 中多个 Query 头共享一组 KV 头。

```python
kv_cache_size = 2 × decoder_layers × num_key_value_heads × head_dim
```

其中：
- `2`: 分别存储 Key 和 Value
- `decoder_layers`: 解码器层数
- `num_key_value_heads`: KV 头数 (小于等于 num_attention_heads)
- `head_dim`: 每个头的维度 (hidden_size)

### TP 并行切分

GQA 的 KV Cache 同样按 KV 头数切分：

```python
kv_cache_per_gpu = kv_cache_size // tp_size
```

每个 GPU 存储 `num_key_value_heads // tp_size` 个 KV 头。

---

## MLA (Multi-head Latent Attention)

### 计算逻辑

MLA 使用低秩压缩表示 KV，不再是多头形式。

```python
kv_cache_size = decoder_layers × (kv_lora_rank + qk_rope_head_dim)
```

其中：
- `decoder_layers`: 解码器层数
- `kv_lora_rank`: KV 低秩压缩维度
- `qk_rope_head_dim`: RoPE 位置编码维度

**注意**: MLA 不需要乘以 2，因为 `kv_lora_rank` 已经包含了压缩后的 K 和 V 信息。

### TP 并行切分

MLA 的 KV Cache **不切分**，所有 TP rank 保存相同的压缩 KV Cache：

```python
kv_cache_per_gpu = kv_cache_size  # 与总大小相同
```

### 为什么 MLA 不切分 KV Cache？

1. **低秩压缩**: KV 被压缩到低维 latent 空间 (`kv_lora_rank` 通常很小，如 512)
2. **解耦设计**:
   - `C_KV`: 压缩的 KV 表示（所有头共享）
   - `K_R`: RoPE 部分（位置信息，所有头共享）
   - `W_KR`, `W_KC`, `W_VC`: 投影矩阵按 TP 切分
3. **TP 切分的是投影矩阵，不是 KV Cache**:
   - Query 的头按 TP 切分
   - 投影矩阵权重按 TP 切分
   - 但 KV Cache 本身存储的是压缩表示

---

## 代码实现

### 文件位置

- `src/arch/kvcache/kvcache.py`

### 函数定义

```python
# MHA/GQA KV Cache 计算
def mha_gqa_kvcache(config: ModelConfig, kvcache_dtype: DataType) -> int:
    """计算所有层的 MHA/GQA KV Cache 大小"""
    decoder_layers = config.num_hidden_layers
    number_of_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size

    kv_cache_size = 2 * decoder_layers * number_of_kv_heads * head_dim
    return kv_cache_size * kvcache_dtype.value


# MHA/GQA TP 切分
def mha_gqa_kvcache_per_gpu(
    config: ModelConfig,
    kvcache_dtype: DataType,
    tp_size: int
) -> int:
    """计算单个 GPU 的 MHA/GQA KV Cache 大小"""
    return mha_gqa_kvcache(config, kvcache_dtype) // tp_size


# MLA KV Cache 计算
def mla_kvcache(config: ModelConfig, kvcache_dtype: DataType) -> int:
    """计算所有层的 MLA KV Cache 大小"""
    decoder_layers = config.num_hidden_layers
    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim

    kv_cache_size = decoder_layers * (kv_lora_rank + qk_rope_head_dim)
    return kv_cache_size * kvcache_dtype.value


# MLA TP 切分（不切分）
def mla_kvcache_per_gpu(
    config: ModelConfig,
    kvcache_dtype: DataType,
    tp_size: int
) -> int:
    """计算单个 GPU 的 MLA KV Cache 大小"""
    # MLA KV Cache 不切分，所有 TP rank 保存相同数据
    return mla_kvcache(config, kvcache_dtype)
```

---

## 对比总结

| 特性 | MHA | GQA | MLA |
|------|-----|-----|-----|
| KV 表示 | 多头独立 | 分组共享 | 低秩压缩 |
| 公式 | `2 × L × H × D` | `2 × L × G × D` | `L × (R + P)` |
| TP 切分 | 按头切分 | 按头切分 | **不切分** |
| 切分维度 | `H // tp_size` | `G // tp_size` | 完整保留 |
| 内存效率 | 低 | 中 | **高** |

其中：
- `L`: decoder_layers
- `H`: num_attention_heads
- `G`: num_key_value_heads
- `D`: head_dim (hidden_size)
- `R`: kv_lora_rank
- `P`: qk_rope_head_dim
