"""
CraftLoRA Standard Inference

Standard inference pipeline for content-style composition using trained
CraftLoRA weights. Supports:
  - Content-only, style-only, or combined content+style LoRA
  - Token-based activation via <c>/<s> in prompt
  - Configurable gamma_c / gamma_s for continuous strength control
"""

import argparse
import os
import logging

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from craftlora_utils import (
    CONTENT_LAYERS, STYLE_LAYERS,
    detect_tokens, aggregate_lora_weights,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CraftLoRA Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt with <c>/<s> tokens")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generated images")
    parser.add_argument("--content_lora", type=str, default=None, help="Path to content LoRA weights")
    parser.add_argument("--style_lora", type=str, default=None, help="Path to style LoRA weights")
    parser.add_argument("--gamma_c", type=float, default=None,
                        help="Content strength [0,1]. Auto-detected from <c> token if not set.")
    parser.add_argument("--gamma_s", type=float, default=None,
                        help="Style strength [0,1]. Auto-detected from <s> token if not set.")
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--vae_path", type=str, default=None, help="Path to local VAE model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local SDXL model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Detect <c>/<s> tokens to determine activation scalars (Contribution 2)
    tokens = detect_tokens(args.prompt)
    gamma_c = args.gamma_c if args.gamma_c is not None else tokens['gamma_c']
    gamma_s = args.gamma_s if args.gamma_s is not None else tokens['gamma_s']
    logger.info(f"Activation scalars: gamma_c={gamma_c}, gamma_s={gamma_s}")

    # Load VAE
    if args.vae_path and os.path.exists(args.vae_path):
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Load pipeline
    model_id = args.model_path if args.model_path and os.path.exists(args.model_path) \
        else "stabilityai/stable-diffusion-xl-base-1.0"
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.float16
    ).to("cuda")

    # Load LoRA weights
    content_sd = {}
    if args.content_lora is not None:
        logger.info(f"Loading content LoRA from: {args.content_lora}")
        content_sd, _ = pipeline.lora_state_dict(args.content_lora)

    style_sd = {}
    if args.style_lora is not None:
        logger.info(f"Loading style LoRA from: {args.style_lora}")
        style_sd, _ = pipeline.lora_state_dict(args.style_lora)

    # Aggregate with disjoint layer filtering (Eq.8)
    merged_lora = aggregate_lora_weights(content_sd, style_sd, gamma_c, gamma_s)
    logger.info(f"Aggregated LoRA: {len(merged_lora)} keys (gamma_c={gamma_c}, gamma_s={gamma_s})")

    pipeline.load_lora_into_unet(merged_lora, None, pipeline.unet)

    # Generate
    logger.info(f"Generating with prompt: {args.prompt}")
    images = pipeline(args.prompt, num_images_per_prompt=args.num_images_per_prompt).images

    os.makedirs(args.output_path, exist_ok=True)
    safe_prompt = args.prompt[:30].replace(" ", "_")
    for i, img in enumerate(images):
        save_path = f'{args.output_path}/image_{i}_{safe_prompt}.jpg'
        img.save(save_path)
        logger.info(f"Saved: {save_path}")
