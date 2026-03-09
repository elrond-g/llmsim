from src.arch.config import ForwardMode, MiniMaxM2Config
from src.arch.kvcache.kvcache import mha_gqa_kvcache, mha_gqa_kvcache_per_gpu
from src.arch.models_arch.base_model_arch import BaseModelArch
from src.arch.op.op_register import create_operator
from src.arch.op.operator_base import DataType, OperatorIO, OperatorMetadata, Tensor


class MiniMaxM2Arch(BaseModelArch):
    """MiniMax M2 Model Architecture (e.g., MiniMax-M2.5)

    Architecture features:
    - GQA attention (48 Q heads, 8 KV heads, head_dim=128)
    - All layers are MoE (256 experts, top-8, no shared expert)
    - Multi-Token Prediction with num_mtp_modules extra layers
    - FP8 quantization (modeled as INT8)
    """

    def build_operators(self) -> None:
        mc = self.model_config
        sc = self.schedule_config

        if not isinstance(mc, MiniMaxM2Config):
            raise ValueError(
                f"MiniMaxM2Arch requires MiniMaxM2Config, but got {type(mc)}"
            )

        # MTP adds num_mtp_modules * mtp_transformer_layers extra layers
        num_mtp_extra = (
            mc.num_mtp_modules * mc.mtp_transformer_layers if sc.is_mtp else 0
        )
        num_layers = mc.num_hidden_layers + num_mtp_extra

        # All layers use the same attention + MoE structure (no dense-only layers)
        self._build_attention_operators(num_layers)
        self._build_moe_operators(num_layers)

        if sc.deepep:
            self._build_deepep_operators(num_layers)

    def _build_attention_operators(self, num_layers: int) -> None:
        """Build GQA attention operators"""
        mc = self.model_config
        sc = self.schedule_config

        assert mc.num_attention_heads % sc.tp_size == 0
        if mc.num_key_value_heads >= sc.tp_size:
            assert mc.num_key_value_heads % sc.tp_size == 0
        else:
            assert sc.tp_size % mc.num_key_value_heads == 0

        num_heads_per_rank = mc.num_attention_heads // sc.tp_size
        kv_heads_per_rank = max(1, mc.num_key_value_heads // sc.tp_size)
        seq_len = self.get_seq_length()
        head_dim = mc.head_dim

        # QKV projection: hidden_size -> (Q_heads + 2*KV_heads) * head_dim
        qkv_proj_metadata = OperatorMetadata(
            name="qkv_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(
                    seq_len,
                    (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim,
                ),
                weight_shape=Tensor(
                    mc.hidden_size,
                    (num_heads_per_rank + kv_heads_per_rank * 2) * head_dim,
                ),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", qkv_proj_metadata))

        # Output projection: Q_heads * head_dim -> hidden_size
        o_proj_metadata = OperatorMetadata(
            name="o_proj",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, num_heads_per_rank * head_dim),
                output_shape=Tensor(seq_len, mc.hidden_size),
                weight_shape=Tensor(num_heads_per_rank * head_dim, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", o_proj_metadata))

        # TP AllReduce after output projection (if TP > 1)
        if sc.tp_size > 1:
            reduce_bandwidth = 85.0 if sc.mode == ForwardMode.EXTEND else 22.64
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

        # Attention core: QK and QKV matmuls
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
            num_layers=num_layers,
        )
        attn_operators.append(create_operator("attention", qk_metadata, mc.attention_type))

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
            num_layers=num_layers,
        )
        attn_operators.append(create_operator("attention", qkv_metadata, mc.attention_type))

        self._add_attention_operator("attention", attn_operators)

    def _build_moe_operators(self, num_layers: int) -> None:
        """Build MoE operators (all layers, no shared expert)"""
        mc = self.model_config
        sc = self.schedule_config

        seq_len = self.get_seq_length()
        assert mc.num_local_experts % sc.ep_size == 0
        experts_per_rank = mc.num_local_experts // sc.ep_size

        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen
        else:
            L = sc.batch_size

        assert L // sc.tp_size * mc.num_experts_per_tok % experts_per_rank == 0
        L_per_rank = L // sc.tp_size * mc.num_experts_per_tok // experts_per_rank

        # intermediate_size in config is the per-expert FFN size
        moe_intermediate_size = mc.intermediate_size
        if not sc.deepep:
            moe_intermediate_size = moe_intermediate_size // sc.tp_size

        # Gate projection (router)
        moe_gate_metadata = OperatorMetadata(
            name="moe_gate",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(seq_len, mc.hidden_size),
                output_shape=Tensor(seq_len, mc.num_local_experts),
                weight_shape=Tensor(mc.hidden_size, mc.num_local_experts),
                input_dtype=DataType.FP32,
                output_dtype=DataType.FP32,
                weight_dtype=DataType.FP32,
            ),
            batch_size=1,
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", moe_gate_metadata))

        # Expert up/gate projection (SwiGLU: gate + up fused)
        moe_up_metadata = OperatorMetadata(
            name="moe_up",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.hidden_size),
                output_shape=Tensor(L_per_rank, 2 * mc.intermediate_size),
                weight_shape=Tensor(mc.hidden_size, 2 * mc.intermediate_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", moe_up_metadata))

        # Expert down projection
        moe_down_metadata = OperatorMetadata(
            name="moe_down",
            op_type="matmul",
            io_config=OperatorIO(
                input_shape=Tensor(L_per_rank, mc.intermediate_size),
                output_shape=Tensor(L_per_rank, mc.hidden_size),
                weight_shape=Tensor(mc.intermediate_size, mc.hidden_size),
                input_dtype=DataType.INT8,
                output_dtype=DataType.BF16,
                weight_dtype=DataType.INT8,
            ),
            batch_size=int(experts_per_rank),
            num_layers=num_layers,
        )
        self._add_operator(create_operator("matmul", moe_down_metadata))

        # No shared expert (shared_intermediate_size == 0)

    def _build_deepep_operators(self, num_layers: int) -> None:
        """Build Deep-EP dispatch/combine transfer operators"""
        mc = self.model_config
        sc = self.schedule_config

        if sc.mode == ForwardMode.EXTEND:
            L = sc.max_seqlen // sc.tp_size
            dispatch_bandwidth = 85.0  # GB/s
            combine_bandwidth = 85.0
        else:
            L = sc.batch_size // sc.tp_size
            dispatch_bandwidth = 18.58  # GB/s
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
            num_layers=num_layers,
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
            num_layers=num_layers,
        )
        combine_op = create_operator("transfer", combine_metadata)
        combine_op._bandwidth_gb_s = combine_bandwidth
        self._add_transfer_operator(combine_op)

    def get_kv_cache(self):
        return mha_gqa_kvcache(self.model_config, DataType.BF16)

    def get_kv_cache_per_gpu(self):
        return mha_gqa_kvcache_per_gpu(
            self.model_config, DataType.BF16, self.schedule_config.tp_size
        )
