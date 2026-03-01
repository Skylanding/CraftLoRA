#!/bin/bash
# CraftLoRA Training Script
# Usage: bash scripts/train.sh

# 1. Rank-Limited Backbone Fine-Tuning (Contribution 1)
python train_backbone.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --content_data_dir="<path/to/10_content_refs>" \
    --style_data_dir="<path/to/10_style_refs>" \
    --output_dir="output/backbone" \
    --r_max=64 --r_min=4 \
    --max_train_steps=1000 \
    --learning_rate=1e-5 \
    --seed=42

# 2. Content LoRA Training on W_init (Contribution 2)
# accelerate launch train_craftlora_sdxl.py \
#     --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
#     --pretrained_backbone_path="output/backbone/unet_w_init" \
#     --instance_data_dir="<path/to/content_images>" \
#     --output_dir="output/content/<subject>" \
#     --instance_prompt="A <subject> <c>" \
#     --resolution=512 \
#     --rank=64 \
#     --train_batch_size=1 \
#     --learning_rate=1e-5 \
#     --max_train_steps=1000 \
#     --checkpointing_steps=500 \
#     --seed=0 \
#     --gradient_checkpointing \
#     --use_8bit_adam \
#     --mixed_precision="fp16"

# 3. Style LoRA Training on W_init (Contribution 2)
# accelerate launch train_craftlora_sdxl.py \
#     --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
#     --pretrained_backbone_path="output/backbone/unet_w_init" \
#     --instance_data_dir="<path/to/style_images>" \
#     --output_dir="output/style/<style_name>" \
#     --instance_prompt="In <style_name> style <s>" \
#     --resolution=512 \
#     --rank=64 \
#     --train_batch_size=1 \
#     --learning_rate=1e-5 \
#     --max_train_steps=1000 \
#     --checkpointing_steps=500 \
#     --seed=0 \
#     --gradient_checkpointing \
#     --use_8bit_adam \
#     --mixed_precision="fp16"
