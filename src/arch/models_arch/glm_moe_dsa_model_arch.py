from src.arch.config import ForwardMode, GlmMoeDsaConfig
from src.arch.kvcache.kvcache import mla_kvcache, mla_kvcache_per_gpu
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.op.op_register import create_operator
from src.arch.op.operator_base import DataType, OperatorIO, OperatorMetadata, Tensor


class GlmMoeDsaArch(BaseModelArch):
    """GLM MoE DSA Model Architecture (e.g., GLM-5)"""

    def build_operators(self) -> None:
        """Build operators for GLM MoE DSA model"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, GlmMoeDsaConfig):
            raise ValueError(
                f"MoE DSA architecture requires GlmMoeDsaConfig, but got {type(mc)}"
            )

        mtp_layers = self._get_mtp_layers()
        total_layers = mc.num_hidden_layers + mtp_layers

        self._build_attention_operators(total_layers)

        dense_layers = min(getattr(mc, "first_k_dense_replace", 0), total_layers)
        if dense_layers > 0:
            self._build_dense_operators(dense_layers)

        moe_layers = self._get_moe_layer_count(total_layers, dense_layers)
        if moe_layers > 0:
            self._build_moe_operators(moe_layers)
            if sc.deepep:
                self._build_deepep_operators(moe_layers)

    def _get_mtp_layers(self) -> int:
        if not self.schedule_config.is_mtp:
            return 0
        return int(getattr(self.model_config, "num_nextn_predict_layers", 1))

    def _get_moe_layer_count(self, total_layers: int, dense_layers: int) -> int:
        remaining_layers = max(0, total_layers - dense_layers)
        moe_layer_freq = max(1, int(getattr(self.model_config, "moe_layer_freq", 1)))
        if moe_layer_freq <= 1:
            return remaining_layers
        return (remaining_layers + moe_layer_freq - 1) // moe_layer_freq

    def _get_attention_k_len(self) -> int:
        sc = self.schedule_config
        mc = self.model_config
        k_len = sc.max_seqlen
        index_topk = int(getattr(mc, "index_topk", 0) or 0)
        if index_topk > 0:
            k_len = min(k_len, index_topk)
        return k_len

    def _build_attention_operators(self, num_layers: int) -> None:
        """Build MLA-style attention operators with DSA top-k"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, GlmMoeDsaConfig):
            raise ValueError("Expected GlmMoeDsaConfig")

        assert mc.num_attention_heads % sc.tp_size == 0
        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        seq_len = self.get_seq_length()

        qk_nope_head_dim = int(getattr(mc, "qk_nope_head_dim", 0) or 0)
        qk_rope_head_dim = int(getattr(mc, "qk_rope_head_dim", 0) or 0)
        if qk_nope_head_dim or qk_rope_head_dim:
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        else:
            qk_head_dim = int(
                getattr(mc, "qk_head_dim", mc.hidden_size // mc.num_attention_heads)
            )
            qk_nope_head_dim = qk_head_dim
            qk_rope_head_dim = 0
        v_head_dim = int(getattr(mc, "v_head_dim", qk_head_dim))

        q_a_kv_a_metadata = OperatorMetadata(
            name="q_a_kv_a",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(
                    seq_len, mc.q_lora_rank + mc.kv_lora_rank + qk_rope_head_dim
                ),
                weight_shape=Tensor(
                    mc.hidden_size,
                    mc.q_lora_rank + mc.kv_lora_rank + qk_rope_head_dim,
                ),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", q_a_kv_a_metadata))

        q_b_metadata = OperatorMetadata(
            name="q_b",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.q_lora_rank),
                output_shape=Tensor(seq_len, num_heads_per_rank * qk_head_dim),
                weight_shape=Tensor(mc.q_lora_rank, num_heads_per_rank * qk_head_dim),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", q_b_metadata))

        if sc.mode == ForwardMode.EXTEND:
            kv_b_metadata = OperatorMetadata(
                name="kv_b",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.kv_lora_rank),
                    output_shape=Tensor(
                        seq_len, num_heads_per_rank * (v_head_dim + qk_nope_head_dim)
                    ),
                    weight_shape=Tensor(
                        mc.kv_lora_rank,
                        num_heads_per_rank * (v_head_dim + qk_nope_head_dim),
                    ),
                    input_dtype=DataType.INT8,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.INT8,
                ),
                batch_size=1,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", kv_b_metadata))

        if sc.mode == ForwardMode.DECODE:
            q_absorb_metadata = OperatorMetadata(
                name="q_absorb",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, qk_nope_head_dim),
                    output_shape=Tensor(seq_len, mc.kv_lora_rank),
                    weight_shape=Tensor(qk_nope_head_dim, mc.kv_lora_rank),
                    input_dtype=DataType.FP32,
                    output_dtype=DataType.FP32,
                    weight_dtype=DataType.FP32,
                ),
                batch_size=num_heads_per_rank,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", q_absorb_metadata))

            o_absorb_metadata = OperatorMetadata(
                name="o_absorb",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.kv_lora_rank),
                    output_shape=Tensor(seq_len, v_head_dim),
                    weight_shape=Tensor(mc.kv_lora_rank, v_head_dim),
                    input_dtype=DataType.FP32,
                    output_dtype=DataType.FP32,
                    weight_dtype=DataType.FP32,
                ),
                batch_size=num_heads_per_rank,
                num_layers=num_layers,
            )
            self._add_operator(create_operator("matmul", o_absorb_metadata))

        o_proj_metadata = OperatorMetadata(
            name="o_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, num_heads_per_rank * v_head_dim),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(num_heads_per_rank * v_head_dim, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", o_proj_metadata))

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

        attn_k_len = self._get_attention_k_len()
        attn_operators = []

        qk_nope_metadata = OperatorMetadata(
            name="qk_nope",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, qk_nope_head_dim),
                output_shape=Tensor(seq_len, attn_k_len),
                weight_shape=Tensor(qk_nope_head_dim, attn_k_len),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        attn_operators.append(
            create_operator("attention", qk_nope_metadata, mc.attention_type)
        )

        if qk_rope_head_dim > 0:
            qk_rope_metadata = OperatorMetadata(
                name="qk_rope",
                op_type="attention",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, qk_rope_head_dim),
                    output_shape=Tensor(seq_len, attn_k_len),
                    weight_shape=Tensor(qk_rope_head_dim, attn_k_len),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.BF16,
                ),
                batch_size=num_heads_per_rank,
                num_layers=num_layers,
            )
            attn_operators.append(
                create_operator("attention", qk_rope_metadata, mc.attention_type)
            )

        qkv_metadata = OperatorMetadata(
            name="qkv",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, attn_k_len),
                output_shape=Tensor(seq_len, v_head_dim),
                weight_shape=Tensor(attn_k_len, v_head_dim),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=num_heads_per_rank,
            num_layers=num_layers,
        )
        attn_operators.append(
            create_operator("attention", qkv_metadata, mc.attention_type)
        )
        self._add_attention_operator("attention", attn_operators)

        self._build_index_attention_operators(num_layers)

    def _build_index_attention_operators(self, num_layers: int) -> None:
        mc = self.model_config
        sc = self.schedule_config

        index_heads = int(getattr(mc, "index_n_heads", 0) or 0)
        index_head_dim = int(getattr(mc, "index_head_dim", 0) or 0)
        if index_heads <= 0 or index_head_dim <= 0:
            return

        if index_heads >= sc.tp_size:
            assert index_heads % sc.tp_size == 0
        index_heads_per_rank = max(1, index_heads // sc.tp_size)
        seq_len = self.get_seq_length()

        index_qk_proj_metadata = OperatorMetadata(
            name="index_qk_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, index_heads_per_rank * index_head_dim * 2),
                weight_shape=Tensor(
                    mc.hidden_size, index_heads_per_rank * index_head_dim * 2
                ),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", index_qk_proj_metadata))

        index_qk_metadata = OperatorMetadata(
            name="index_qk",
            op_type="attention",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, index_head_dim),
                output_shape=Tensor(seq_len, sc.max_seqlen),
                weight_shape=Tensor(index_head_dim, sc.max_seqlen),
                input_dtype=DataType.BF16,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.BF16,
            ),
            batch_size=index_heads_per_rank,
            num_layers=num_layers,
        )
        index_operators = [
            create_operator("attention", index_qk_metadata, mc.attention_type)
        ]
        self._add_attention_operator("index_attention", index_operators)

    def _build_dense_operators(self, num_dense_layers: int) -> None:
        """Build dense layer operators for the first K layers"""
        mc = self.model_config
        sc = self.schedule_config

        seq_len = self.get_seq_length()
        assert mc.intermediate_size % sc.tp_size == 0
        intermediate_size = mc.intermediate_size
        if not sc.enable_moe_dense_fully_dp:
            intermediate_size = intermediate_size // sc.tp_size

        gate_up_metadata = OperatorMetadata(
            name="dense_gate_up_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, 2 * intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_dense_layers,
        )
        self._add_operator(create_operator("matmul", gate_up_metadata))

        down_metadata = OperatorMetadata(
            name="dense_down_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, intermediate_size),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_dense_layers,
        )
        self._add_operator(create_operator("matmul", down_metadata))

        if sc.tp_size > 1 and not sc.enable_moe_dense_fully_dp:
            if sc.mode == ForwardMode.EXTEND:
                reduce_bandwidth = 85.0
            else:
                reduce_bandwidth = 22.64

            all_reduce_metadata = OperatorMetadata(
                name="dense_all_reduce",
                op_type="transfer",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(0, 0),
                    input_dtype=DataType.BF16,
                    output_dtype=DataType.BF16,
                ),
                batch_size=1,
                num_layers=num_dense_layers,
            )
            all_reduce_op = create_operator("transfer", all_reduce_metadata)
            all_reduce_op._bandwidth_gb_s = reduce_bandwidth
            self._add_transfer_operator(all_reduce_op)

    def _build_moe_operators(self, num_moe_layers: int) -> None:
        """Build MoE operators with shared expert support"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, GlmMoeDsaConfig):
            raise ValueError("Expected GlmMoeDsaConfig")

        num_experts = int(getattr(mc, "n_routed_experts", 0) or 0)
        if num_experts <= 0:
            num_experts = int(getattr(mc, "num_experts", 0) or 0)
        if num_experts <= 0:
            raise ValueError("n_routed_experts must be positive for MoE operators")

        seq_len = self.get_seq_length()
        assert num_experts % sc.ep_size == 0
        experts_per_rank = num_experts // sc.ep_size

        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen
        else:
            L = sc.batch_size

        assert L // sc.tp_size * mc.num_experts_per_tok % experts_per_rank == 0
        L_per_rank = L // sc.tp_size * mc.num_experts_per_tok // experts_per_rank

        shared_experts = int(getattr(mc, "n_shared_experts", 0) or 0)

        moe_intermediate_size = mc.moe_intermediate_size
        if not sc.deepep:
            moe_intermediate_size = moe_intermediate_size // sc.tp_size

        moe_gate_metadata = OperatorMetadata(
            name="moe_gate",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, num_experts),
                weight_shape=Tensor(mc.hidden_size, num_experts),
                input_dtype=DataType.FP32,
                output_dtype=DataType.FP32,
                weight_dtype=DataType.FP32,
            ),
            batch_size=1,
            num_layers=num_moe_layers,
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
            num_layers=num_moe_layers,
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
            num_layers=num_moe_layers,
        )
        self._add_operator(create_operator("matmul", moe_down_metadata))

        if shared_experts > 0:
            share_up_metadata = OperatorMetadata(
                name="share_up",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, mc.hidden_size),
                    output_shape=Tensor(seq_len, 2 * moe_intermediate_size),
                    weight_shape=Tensor(mc.hidden_size, 2 * moe_intermediate_size),
                    input_dtype=DataType.INT8,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.INT8,
                ),
                batch_size=shared_experts,
                num_layers=num_moe_layers,
            )
            self._add_operator(create_operator("matmul", share_up_metadata))

            share_down_metadata = OperatorMetadata(
                name="share_down",
                op_type="matmul",
                io_config=OperatorIO(
                    input_shape=Tensor(seq_len, moe_intermediate_size),
                    output_shape=Tensor(seq_len, mc.hidden_size),
                    weight_shape=Tensor(moe_intermediate_size, mc.hidden_size),
                    input_dtype=DataType.INT8,
                    output_dtype=DataType.BF16,
                    weight_dtype=DataType.INT8,
                ),
                batch_size=shared_experts,
                num_layers=num_moe_layers,
            )
            self._add_operator(create_operator("matmul", share_down_metadata))

    def _build_deepep_operators(self, num_moe_layers: int) -> None:
        """Build Deep-EP transfer operators"""
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, GlmMoeDsaConfig):
            raise ValueError("Expected GlmMoeDsaConfig")

        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen // sc.tp_size
            dispatch_bandwidth = 85.0
            combine_bandwidth = 85.0
        else:
            L = sc.batch_size // sc.tp_size
            dispatch_bandwidth = 18.58
            combine_bandwidth = 22.64

        dispatch_metadata = OperatorMetadata(
            name="dispatch",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(L, mc.hidden_size),
                output_shape=Tensor(L, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.INT8,
            ),
            batch_size=mc.num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        dispatch_op = create_operator("transfer", dispatch_metadata)
        dispatch_op._bandwidth_gb_s = dispatch_bandwidth
        self._add_transfer_operator(dispatch_op)

        combine_metadata = OperatorMetadata(
            name="combine",
            op_type="transfer",
            io_config=OperatorIO(
                input_shape=Tensor(L, mc.hidden_size),
                output_shape=Tensor(L, mc.hidden_size),
                weight_shape=Tensor(0, 0),
                input_dtype=DataType.BF16,
            ),
            batch_size=mc.num_experts_per_tok,
            num_layers=num_moe_layers,
        )
        combine_op = create_operator("transfer", combine_metadata)
        combine_op._bandwidth_gb_s = combine_bandwidth
        self._add_transfer_operator(combine_op)

    def get_kv_cache(self):
        return mla_kvcache(self.model_config, DataType.INT8)

    def get_kv_cache_per_gpu(self):
        return mla_kvcache_per_gpu(
            self.model_config, DataType.INT8, self.schedule_config.tp_size
        )
