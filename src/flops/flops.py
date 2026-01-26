from src.config.model_config import ModelConfig, HybridAttnConfig
from src.server_args import ServerArgs


def gemm_flops(m, n, k):
    return 2.0 * m * n * k


def get_mha_gflops(config: ModelConfig, server_args: ServerArgs, bs: int, avg_context_len: int):
    tp_size = max(1, server_args.tp_size)
    attn_cfg = config.attn_config if hasattr(config, "attn_config") else config
    if isinstance(attn_cfg, HybridAttnConfig):
        attn_cfg = attn_cfg.full_attn_config

    q_proj = gemm_flops(
        bs,
        config.hidden_size,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.head_dim,
    )
    k_proj = gemm_flops(
        bs,
        config.hidden_size,
        (attn_cfg.num_key_value_heads // tp_size) * attn_cfg.head_dim,
    )
    v_proj = gemm_flops(
        bs,
        config.hidden_size,
        (attn_cfg.num_key_value_heads // tp_size) * attn_cfg.head_dim,
    )
    o_proj = gemm_flops(
        bs,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.head_dim,
        config.hidden_size,
    )
    attn_core = gemm_flops(
        bs, (attn_cfg.num_attention_heads // tp_size) * attn_cfg.head_dim, avg_context_len
    ) + gemm_flops(
        bs, avg_context_len, (attn_cfg.num_attention_heads // tp_size) * attn_cfg.head_dim
    )
    return attn_core / 1e9, (q_proj + k_proj + v_proj + o_proj) / 1e9


def get_mla_absorb_gflops(config: ModelConfig, server_args: ServerArgs, bs: int, avg_context_len: int):
    tp_size = max(1, server_args.tp_size)
    attn_cfg = config.attn_config if hasattr(config, "attn_config") else config
    if isinstance(attn_cfg, HybridAttnConfig):
        attn_cfg = attn_cfg.full_attn_config

    q_down_proj = gemm_flops(bs, config.hidden_size, attn_cfg.q_lora_rank)
    q_up_proj = gemm_flops(
        bs,
        attn_cfg.q_lora_rank,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.qk_head_dim,
    )

    kv_down_proj = gemm_flops(
        bs, config.hidden_size, attn_cfg.kv_lora_rank + attn_cfg.qk_rope_head_dim
    )

    bmm_q_wk = (attn_cfg.num_attention_heads // tp_size) * gemm_flops(
        bs, attn_cfg.qk_nope_head_dim, attn_cfg.kv_lora_rank
    )
    bmm_o_wv = (attn_cfg.num_attention_heads // tp_size) * gemm_flops(
        bs, attn_cfg.kv_lora_rank, attn_cfg.v_head_dim
    )

    o_proj = gemm_flops(
        bs,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.v_head_dim,
        config.hidden_size,
    )

    attn_core = gemm_flops(
        bs,
        (attn_cfg.num_attention_heads // tp_size)
        * (attn_cfg.kv_lora_rank + attn_cfg.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs,
        avg_context_len,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.kv_lora_rank,
    )

    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj) / 1e9,
    )


def get_gqla_absorb_gflops(config: ModelConfig, server_args: ServerArgs, bs: int, avg_context_len: int):
    tp_size = max(1, server_args.tp_size)
    attn_cfg = config.attn_config if hasattr(config, "attn_config") else config
    if isinstance(attn_cfg, HybridAttnConfig):
        attn_cfg = attn_cfg.full_attn_config

    q_down_proj = gemm_flops(bs, config.hidden_size, attn_cfg.q_lora_rank)
    q_up_proj = gemm_flops(
        bs,
        attn_cfg.q_lora_rank,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.qk_head_dim,
    )

    kv_down_proj = gemm_flops(
        bs, config.hidden_size, attn_cfg.kv_lora_rank + attn_cfg.qk_rope_head_dim
    )

    # GQLA specific logic (preserving the /2 factor if intentional)
    bmm_q_wk = (
        2
        * (attn_cfg.num_attention_heads // tp_size)
        / 2
        * gemm_flops(bs, attn_cfg.qk_nope_head_dim, attn_cfg.kv_lora_rank / 2)
    )
    bmm_o_wv = (
        2
        * (attn_cfg.num_attention_heads // tp_size)
        / 2
        * gemm_flops(bs, attn_cfg.kv_lora_rank / 2, attn_cfg.v_head_dim)
    )

    o_proj = gemm_flops(
        bs,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.v_head_dim,
        config.hidden_size,
    )

    attn_core = gemm_flops(
        bs,
        (attn_cfg.num_attention_heads // tp_size)
        * (attn_cfg.kv_lora_rank / 2 + attn_cfg.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs,
        avg_context_len,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.kv_lora_rank / 2,
    )

    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj) / 1e9,
    )


def get_mla_noabsorb_gflops(config: ModelConfig, server_args: ServerArgs, bs: int, avg_context_len: int):
    tp_size = max(1, server_args.tp_size)
    attn_cfg = config.attn_config if hasattr(config, "attn_config") else config
    if isinstance(attn_cfg, HybridAttnConfig):
        attn_cfg = attn_cfg.full_attn_config

    q_down_proj = gemm_flops(bs, config.hidden_size, attn_cfg.q_lora_rank)
    q_up_proj = gemm_flops(
        bs,
        attn_cfg.q_lora_rank,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.qk_head_dim,
    )

    kv_down_proj = gemm_flops(
        bs, config.hidden_size, attn_cfg.kv_lora_rank + attn_cfg.qk_rope_head_dim
    )
    kv_up_proj = gemm_flops(
        bs,
        attn_cfg.kv_lora_rank,
        (attn_cfg.num_attention_heads // tp_size)
        * (attn_cfg.v_head_dim + attn_cfg.qk_nope_head_dim),
    )

    o_proj = gemm_flops(
        bs,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.v_head_dim,
        config.hidden_size,
    )

    attn_core = gemm_flops(
        bs,
        (attn_cfg.num_attention_heads // tp_size)
        * (attn_cfg.qk_nope_head_dim + attn_cfg.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs,
        avg_context_len,
        (attn_cfg.num_attention_heads // tp_size) * attn_cfg.v_head_dim,
    )

    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + kv_up_proj + o_proj) / 1e9,
    )


def get_attn_gflops(
    config: ModelConfig, server_args: ServerArgs, avg_context_len: int, absorb=True
):
    # Determine attn type from config
    attn_cfg = config.attn_config
    if isinstance(attn_cfg, HybridAttnConfig):
        # Default to full attn for GFLOPS calculation or handle based on layer?
        # Usually we want the GFLOPS of a specific layer type.
        attn_cfg = attn_cfg.full_attn_config

    from src.config.model_config import MLAConfig, MHAConfig

    if isinstance(attn_cfg, MHAConfig):
        return get_mha_gflops(config, server_args, bs=1, avg_context_len=avg_context_len)
    elif isinstance(attn_cfg, MLAConfig):
        if absorb:
            return get_mla_absorb_gflops(
                config, server_args, bs=1, avg_context_len=avg_context_len
            )
        return get_mla_noabsorb_gflops(
            config, server_args, bs=1, avg_context_len=avg_context_len
        )
    return 0.0, 0.0


def get_moe_gflops(config: ModelConfig, server_args: ServerArgs):
    moe_cfg = config.moe_config
    if not moe_cfg:
        return 0.0

    tp_size = max(1, server_args.tp_size)
    world_size = max(1, server_args.world_size)

    # Shared experts (usually replicated and TP-ed)
    num_shared = moe_cfg.num_shared_experts
    shared_gflops = (
        num_shared
        * 3.0
        * gemm_flops(1, config.hidden_size, moe_cfg.intermediate_size // tp_size)
        / 1e9
    )

    # Routed experts (distributed across EP)
    # Total routed compute in system per token = top_k * 3 * gemm(1, H, I)
    # Per device routed compute (average) = Total / world_size
    # Note: each expert internally might be TP-ed if ep_tp_size > 1
    routed_gflops_total = (
        moe_cfg.num_experts_per_tok
        * 3.0
        * gemm_flops(1, config.hidden_size, moe_cfg.intermediate_size)
        / 1e9
    )
    routed_gflops_per_device = routed_gflops_total / world_size

    return shared_gflops + routed_gflops_per_device
