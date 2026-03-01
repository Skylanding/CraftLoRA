#!/bin/bash
# CraftLoRA Inference Script
# Usage: bash scripts/infer.sh

# 1. Standard Inference: Content + Style LoRA with token routing
python inference.py \
    --prompt="A dog <c> in cartoon style <s>" \
    --content_lora="<path/to/content/pytorch_lora_weights.safetensors>" \
    --style_lora="<path/to/style/pytorch_lora_weights.safetensors>" \
    --output_path="results/combined" \
    --num_images_per_prompt=4

# 2. ACFG Inference (Contribution 3)
# python inference_acfg.py \
#     --prompt="A cat <c> in oil painting style <s>" \
#     --content_lora="<path/to/content/pytorch_lora_weights.safetensors>" \
#     --style_lora="<path/to/style/pytorch_lora_weights.safetensors>" \
#     --output_path="results/acfg" \
#     --guidance_omega=7.5 \
#     --num_images_per_prompt=4
