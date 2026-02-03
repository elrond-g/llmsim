python3 src/main.py --model_path hf_config/qwen3-32B_config.json \
    --max_seqlen 4096 \
    --batch_size 2 \
    --mode extend \
    --tp_size 1 \
    --dp_size 8 \
    --hardware klx_p800 \
    --output_format console