export PYTHONPATH=$PYTHONPATH:.

python3 src/main.py --model_path hf_config/deepseek_671b_r1_config.json \
    --max_seqlen 4096 \
    --batch_size 64 \
    --mode decode \
    --tp_size 1 \
    --dp_size 16 \
    --ep_size 16 \
    --enable_deepep \
    --enable_moe_dense_fully_dp \
    --hardware klx_p800 \
    --output_format console


python3 src/main.py --model_path hf_config/deepseek_671b_r1_config.json \
    --max_seqlen 4096 \
    --batch_size 64 \
    --mode decode \
    --tp_size 1 \
    --dp_size 16 \
    --ep_size 16 \
    --enable_deepep \
    --enable_moe_dense_fully_dp \
    --hardware klx_p800 \
    --output_format excel \
    --output_file metrics/ds_v3_decode_result.xlsx
