#!/bin/bash

# Default values for environment variables
export MODEL_ID="black-forest-labs/FLUX.1-dev"
export ADAPTER_ID="tungdop2/FLUX.1-dev-dreambooth-veronika_512"
export OUTPUT_DIR="output/images"
export STEPS=50
export GUIDANCE_SCALE=3.5
export MAX_SEQUENCE_LENGTH=512
export CONFIG_PATH="webhook.json"
export PROMPTS_FILE="prompts.txt"

# Run the Python script with the exported environment variables
CUDA_VISIBLE_DEVICES=2 \
python3 generate.py \
    --model_id "$MODEL_ID" \
    --adapter_id "$ADAPTER_ID" \
    --output_dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --max_sequence_length "$MAX_SEQUENCE_LENGTH" \
    --config_path "$CONFIG_PATH" \
    --prompts_file "$PROMPTS_FILE"
