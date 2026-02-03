export PYTHONPATH=$PYTHONPATH:.
python3 src/main.py --model_path hf_config/qwen3-32B_config.json \
    --max_seqlen 4096 \
    --batch_size 1 \
    --mode extend \
    --tp_size 8 \
    --dp_size 1 \
    --hardware klx_p800 \
    --output_format console


python3 src/main.py --model_path hf_config/qwen3-32B_config.json \
    --max_seqlen 4096 \
    --batch_size 1 \
    --mode extend \
    --tp_size 8 \
    --dp_size 1 \
    --hardware klx_p800 \
    --output_format excel \
    --output_file metrics/qwen3-32B_prefill.xlsx
