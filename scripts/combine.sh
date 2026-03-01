#!/bin/bash
# CraftLoRA Batch Training Script
# Trains content and style LoRAs for all subjects/styles in a dataset directory.
# Assumes backbone W_init has already been trained via train_backbone.py.
# Usage: bash scripts/combine.sh

BACKBONE_PATH="output/backbone/unet_w_init"
CONTENT_DATASET_DIR="<path/to/content_dataset>"
STYLE_DATASET_DIR="<path/to/style_dataset>"
OUTPUT_BASE="<path/to/output>"

# Train Content LoRAs (one per subject folder)
for folder in "$CONTENT_DATASET_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        folder_name_with_space=$(echo "$folder_name" | sed 's/_/ /g')

        echo "=== Training content LoRA for: $folder_name_with_space ==="

        accelerate launch train_craftlora_sdxl.py \
            --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
            --pretrained_backbone_path="$BACKBONE_PATH" \
            --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
            --instance_data_dir="$folder" \
            --output_dir="$OUTPUT_BASE/content/$folder_name" \
            --instance_prompt="A $folder_name_with_space <c>" \
            --resolution=512 \
            --rank=64 \
            --train_batch_size=1 \
            --learning_rate=1e-5 \
            --max_train_steps=1000 \
            --checkpointing_steps=500 \
            --seed="0" \
            --gradient_checkpointing \
            --use_8bit_adam \
            --mixed_precision="fp16"
    fi
done

# Train Style LoRAs (one per style folder)
for folder in "$STYLE_DATASET_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        folder_name_with_space=$(echo "$folder_name" | sed 's/_/ /g')

        echo "=== Training style LoRA for: $folder_name_with_space ==="

        accelerate launch train_craftlora_sdxl.py \
            --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
            --pretrained_backbone_path="$BACKBONE_PATH" \
            --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
            --instance_data_dir="$folder" \
            --output_dir="$OUTPUT_BASE/style/$folder_name" \
            --instance_prompt="In $folder_name_with_space style <s>" \
            --resolution=512 \
            --rank=64 \
            --train_batch_size=1 \
            --learning_rate=1e-5 \
            --max_train_steps=1000 \
            --checkpointing_steps=500 \
            --seed="0" \
            --gradient_checkpointing \
            --use_8bit_adam \
            --mixed_precision="fp16"
    fi
done

echo "=== Batch training complete ==="
