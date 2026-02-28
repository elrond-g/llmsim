from src.arch.config import ForwardMode, Qwen3_5MoEConfig
from src.arch.kvcache.kvcache import mha_gqa_kvcache, mha_gqa_kvcache_per_gpu
from src.arch.model_type import AttentionType
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.op.op_register import create_operator
from src.arch.op.operator_base import DataType, OperatorIO, OperatorMetadata, Tensor


class Qwen3_5MoEArch(BaseModelArch):
    """Qwen3.5 MoE Model Architecture (e.g., Qwen3.5-397B-A17B)"""

    def build_operators(self) -> None:
        """Build operators for Qwen3.5 MoE model"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, Qwen3_5MoEConfig):
            raise ValueError(
                f"MoE architecture requires Qwen3_5MoEConfig, but got {type(mc)}"
            )

        mtp_layers = self._get_mtp_layers()
        num_layers = mc.num_hidden_layers + mtp_layers

        self._build_attention_operators(num_layers, mtp_layers)
        self._build_moe_operators(num_layers)

        if sc.deepep:
            self._build_deepep_operators(num_layers)

    def _get_mtp_layers(self) -> int:
        if not self.schedule_config.is_mtp:
            return 0
        return int(getattr(self.model_config, "mtp_num_hidden_layers", 1))

    def _get_attention_layer_counts(
        self, total_layers: int, mtp_layers: int
    ) -> tuple[int, int]:
        mc = self.model_config
        layer_types = getattr(mc, "layer_types", None) or []
        full_layers = sum(1 for layer in layer_types if layer == "full_attention")
        linear_layers = sum(1 for layer in layer_types if layer == "linear_attention")

        if full_layers + linear_layers != mc.num_hidden_layers:
            full_layers = mc.num_hidden_layers
            linear_layers = 0

        if mtp_layers:
            full_layers += mtp_layers

        if full_layers + linear_layers != total_layers:
            full_layers = total_layers
            linear_layers = 0

        return full_layers, linear_layers

    def _build_attention_operators(self, num_layers: int, mtp_layers: int) -> None:
        """Build hybrid attention operators (linear + full attention)"""
        mc = self.model_config
        sc = self.schedule_config

        # Full attention head config
        assert mc.num_attention_heads % sc.tp_size == 0
        if mc.num_key_value_heads > sc.tp_size:
            assert mc.num_key_value_heads % sc.tp_size == 0
        else:
            assert sc.tp_size % mc.num_key_value_heads == 0

        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        kv_heads_per_rank = max(1, mc.num_key_value_heads // sc.tp_size)
        seq_len = self.get_seq_length()
        head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)

        # Linear attention head config
        linear_num_key_heads = getattr(mc, "linear_num_key_heads", 16)
        linear_num_value_heads = getattr(mc, "linear_num_value_heads", 64)
        linear_key_head_dim = getattr(mc, "linear_key_head_dim", 128)
        linear_value_head_dim = getattr(mc, "linear_value_head_dim", 128)

        # TP partitioning for linear attention heads (similar to KV head handling)
        # When heads >= tp_size: require divisibility; when heads < tp_size: replicate
        if linear_num_key_heads >= sc.tp_size:
            assert linear_num_key_heads % sc.tp_size == 0, (
                f"linear_num_key_heads ({linear_num_key_heads}) must be divisible by "
                f"tp_size ({sc.tp_size}) when >= tp_size"
            )
        if linear_num_value_heads >= sc.tp_size:
            assert linear_num_value_heads % sc.tp_size == 0, (
                f"linear_num_value_heads ({linear_num_value_heads}) must be divisible by "
                f"tp_size ({sc.tp_size}) when >= tp_size"
            )
        linear_key_heads_per_rank = max(1, linear_num_key_heads // sc.tp_size)
        linear_value_heads_per_rank = max(1, linear_num_value_heads // sc.tp_size)

        full_layers, linear_layers = self._get_attention_layer_counts(
            num_layers, mtp_layers
        )

        # Build full attention projection operators
        if full_layers:
            full_qkv_proj_metadata = OperatorMetadata(
                name="qkv_proj",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(
                        seq_len, (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim
                    ),
                    weight_shape=Tensor(
                        mc.hidden_size,
                        (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim,
                    ),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=full_layers,
            )
            self._add_operator(create_operator("matmul", full_qkv_proj_metadata))

            full_o_proj_metadata = OperatorMetadata(
                name="o_proj",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, num_heads_per_rank * head_dim),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(num_heads_per_rank * head_dim, mc.hidden_size),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=full_layers,
            )
            self._add_operator(create_operator("matmul", full_o_proj_metadata))

        # Build linear attention projection operators
        if linear_layers:
            # Linear attention Q projection: num_attention_heads * linear_key_head_dim
            # Linear attention K projection: linear_num_key_heads * linear_key_head_dim
            # Linear attention V projection: linear_num_value_heads * linear_value_head_dim
            linear_q_dim = num_heads_per_rank * linear_key_head_dim
            linear_k_dim = linear_key_heads_per_rank * linear_key_head_dim
            linear_v_dim = linear_value_heads_per_rank * linear_value_head_dim

            linear_qkv_proj_metadata = OperatorMetadata(
                name="linear_qkv_proj",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(seq_len, linear_q_dim + linear_k_dim + linear_v_dim),
                    weight_shape=Tensor(
                        mc.hidden_size, linear_q_dim + linear_k_dim + linear_v_dim
                    ),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=linear_layers,
            )
            self._add_operator(create_operator("matmul", linear_qkv_proj_metadata))

            # Linear attention output projection
            linear_o_proj_metadata = OperatorMetadata(
                name="linear_o_proj",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, num_heads_per_rank * linear_value_head_dim),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(
                        num_heads_per_rank * linear_value_head_dim, mc.hidden_size
                    ),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=linear_layers,
            )
            self._add_operator(create_operator("matmul", linear_o_proj_metadata))

        if sc.tp_size > 1:
            if sc.mode == ForwardMode.EXTEND:
                reduce_bandwidth = 85.0
            else:
                reduce_bandwidth = 22.64

            all_reduce_metadata = OperatorMetadata(
                name="attn_all_reduce",
                op_type="transfer",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(0, 0),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=num_layers,
            )
            all_reduce_op = create_operator("transfer", all_reduce_metadata)
            all_reduce_op._bandwidth_gb_s = reduce_bandwidth
            self._add_transfer_operator(all_reduce_op)

        # Build full attention core operators
        if full_layers:
            attn_operators = []
            qk_metadata = OperatorMetadata(
                name="qk",
                op_type="attention",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, head_dim),
                    output_shape=Tensor(seq_len, sc.max_seqlen),
                    weight_shape=Tensor(head_dim, sc.max_seqlen),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=num_heads_per_rank,
                num_layers=full_layers,
            )
            attn_operators.append(
                create_operator("attention", qk_metadata, mc.attention_type)
            )

            qkv_metadata = OperatorMetadata(
                name="qkv",
                op_type="attention",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, sc.max_seqlen),
                    output_shape=Tensor(seq_len, head_dim),
                    weight_shape=Tensor(sc.max_seqlen, head_dim),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=kv_heads_per_rank,
                num_layers=full_layers,
            )
            attn_operators.append(
                create_operator("attention", qkv_metadata, mc.attention_type)
            )

            key = "attention" if linear_layers == 0 else "full_attention"
            self._add_attention_operator(key, attn_operators)

        # Build linear attention core operators with linear_* head dimensions
        if linear_layers:
            linear_operators = []
            # Linear QK: uses linear_key_head_dim, num_heads_per_rank (Q heads same as full attention)
            linear_qk_metadata = OperatorMetadata(
                name="linear_qk",
                op_type="attention",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, linear_key_head_dim),
                    output_shape=Tensor(seq_len, sc.max_seqlen),
                    weight_shape=Tensor(linear_key_head_dim, sc.max_seqlen),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=num_heads_per_rank,
                num_layers=linear_layers,
            )
            linear_operators.append(
                create_operator("attention", linear_qk_metadata, AttentionType.LINEAR)
            )

            # Linear QKV: uses linear_value_head_dim and linear_value_heads_per_rank
            linear_qkv_metadata = OperatorMetadata(
                name="linear_qkv",
                op_type="attention",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, sc.max_seqlen),
                    output_shape=Tensor(seq_len, linear_value_head_dim),
                    weight_shape=Tensor(sc.max_seqlen, linear_value_head_dim),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=linear_value_heads_per_rank,
                num_layers=linear_layers,
            )
            linear_operators.append(
                create_operator("attention", linear_qkv_metadata, AttentionType.LINEAR)
            )

            self._add_attention_operator("linear_attention", linear_operators)

    def _build_moe_operators(self, num_layers: int) -> None:
        """Build MoE operators with shared expert"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, Qwen3_5MoEConfig):
            raise ValueError("Expected Qwen3_5MoEConfig")

        seq_len = self.get_seq_length()
        assert mc.num_experts % sc.ep_size == 0
        experts_per_rank = mc.num_experts // sc.ep_size

        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen
        else:
            L = sc.batch_size

        assert L // sc.tp_size * mc.num_experts_per_tok % experts_per_rank == 0
        L_per_rank = L // sc.tp_size * mc.num_experts_per_tok // experts_per_rank

        shared_intermediate_size = getattr(mc, "shared_expert_intermediate_size", 0)
        if shared_intermediate_size <= 0:
            shared_intermediate_size = mc.moe_intermediate_size
        if not sc.deepep:
            shared_intermediate_size = shared_intermediate_size // sc.tp_size

        moe_gate_metadata = OperatorMetadata(
            name="moe_gate",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, mc.num_experts),
                weight_shape=Tensor(mc.hidden_size, mc.num_experts),
                input_dtype=DataType.FP32,
                output_dtype=DataType.FP32,
                weight_dtype=DataType.FP32,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", moe_gate_metadata))

        moe_up_metadata = OperatorMetadata(
            name="moe_up",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.hidden_size),
                output_shape=Tensor(L_per_rank, 2 * mc.moe_intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * mc.moe_intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", moe_up_metadata))

        moe_down_metadata = OperatorMetadata(
            name="moe_down",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.moe_intermediate_size),
                output_shape=Tensor(L_per_rank, mc.hidden_size),
                weight_shape=Tensor(mc.moe_intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", moe_down_metadata))

        if shared_intermediate_size > 0:
            share_up_metadata = OperatorMetadata(
                name="share_up",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(seq_len, 2 * shared_intermediate_size),
                    weight_shape=Tensor(mc.hidden_size, 2 * shared_intermediate_size),
                    input_dtype=DataType.INT8,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.INT8,
                ),
                batch_size=1,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", share_up_metadata))

            share_down_metadata = OperatorMetadata(
                name="share_down",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, shared_intermediate_size),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(shared_intermediate_size, mc.hidden_size),
                    input_dtype=DataType.INT8,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.INT8,
                ),
                batch_size=1,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", share_down_metadata))

    def _build_deepep_operators(self, num_layers: int) -> None:
        """Build Deep-EP transfer operators"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, Qwen3_5MoEConfig):
            raise ValueError("Expected Qwen3_5MoEConfig")

        seq_len = self.get_seq_length()

        dispatch_metadata = OperatorMetadata(
            name="dispatch",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.FP32,
                output_dtype=DataType.FP32,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        dispatch_op = create_operator("transfer", dispatch_metadata)
        if sc.mode == ForwardMode.EXTEND:
            dispatch_op._bandwidth_gb_s = 100.0
        else:
            dispatch_op._bandwidth_gb_s = 18.58
        self._add_transfer_operator(dispatch_op)

        combine_metadata = OperatorMetadata(
            name="combine",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.FP32,
                output_dtype=DataType.FP32,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        combine_op = create_operator("transfer", combine_metadata)
        if sc.mode == ForwardMode.EXTEND:
            combine_op._bandwidth_gb_s = 100.0
        else:
            combine_op._bandwidth_gb_s = 22.64
        self._add_transfer_operator(combine_op)

    def get_kv_cache(self):
        return mha_gqa_kvcache(self.model_config, DataType.BF16)

    def get_kv_cache_per_gpu(self):
        return mha_gqa_kvcache_per_gpu(
            self.model_config, DataType.BF16, self.schedule_config.tp_size
        )
