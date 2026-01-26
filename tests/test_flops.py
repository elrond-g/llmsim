import os
import pytest
from src.config.model_config import ModelConfig
from src.server_args import ServerArgs
from src.flops.flops import get_attn_gflops, get_moe_gflops

def test_flops_calculation():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "hf_config/qwen3-8B_config.json")
    
    if not os.path.exists(config_path):
        pytest.skip("Config file not found")
        
    config = ModelConfig.from_config_path(config_path)
    
    # Test TP=1
    server_args_tp1 = ServerArgs(config_path=config_path, tp_size=1, world_size=1)
    attn_core_1, attn_proj_1 = get_attn_gflops(config, server_args_tp1, avg_context_len=1024)
    moe_1 = get_moe_gflops(config, server_args_tp1)
    
    print(f"\nTP=1: Attn Core={attn_core_1:.2f} GFLOPS, Attn Proj={attn_proj_1:.2f} GFLOPS, MoE={moe_1:.2f} GFLOPS")
    
    # Test TP=2
    server_args_tp2 = ServerArgs(config_path=config_path, tp_size=2, world_size=2)
    attn_core_2, attn_proj_2 = get_attn_gflops(config, server_args_tp2, avg_context_len=1024)
    moe_2 = get_moe_gflops(config, server_args_tp2)
    
    print(f"TP=2: Attn Core={attn_core_2:.2f} GFLOPS, Attn Proj={attn_proj_2:.2f} GFLOPS, MoE={moe_2:.2f} GFLOPS")
    
    # Assertions for per-device reduction
    assert attn_proj_2 == attn_proj_1 / 2
    assert attn_core_2 == attn_core_1 / 2
    # For DenseMLP (represented as MoE with 1 expert), it should also be / 2
    assert moe_2 == moe_1 / 2

if __name__ == "__main__":
    test_flops_calculation()
