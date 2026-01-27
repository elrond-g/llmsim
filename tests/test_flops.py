import os
import pytest
from src.config.model_config import ModelConfig
from src.server_args import ServerArgs
from src.layers.decode_block import DecoderBlocks

def test_flops_integration():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "hf_config/qwen3-8B_config.json")
    
    if not os.path.exists(config_path):
        pytest.skip("Config file not found")
        
    config = ModelConfig.from_config_path(config_path)
    
    print(f"\nTesting FLOPs integration for: {os.path.basename(config_path)}")
    
    # Test TP=1
    server_args_tp1 = ServerArgs(config_path=config_path, tp_size=1, world_size=1)
    blocks_tp1 = DecoderBlocks(server_args_tp1, config)
    print("\n--- TP=1 Info ---")
    blocks_tp1.print_flops_info(avg_context_len=1024)
    
    # Test TP=2
    server_args_tp2 = ServerArgs(config_path=config_path, tp_size=2, world_size=2)
    blocks_tp2 = DecoderBlocks(server_args_tp2, config)
    print("\n--- TP=2 Info ---")
    blocks_tp2.print_flops_info(avg_context_len=1024)

if __name__ == "__main__":
    test_flops_integration()
