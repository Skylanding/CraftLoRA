"""
CraftLoRA Asymmetric Classifier-Free Guidance (ACFG) Inference (Contribution 3)

Implements Algorithm 2 from the paper:
  - Conditional path: UNet with LoRA adapters (W_cond)
  - Unconditional path: anchored to rank-limited backbone W_init (no LoRA)
  - Timestep-dependent activation scheduling for content/style LoRAs
  - ACFG formula: ε_acfg = (1 + ω) * ε_cond - ω * ε_uncond   (Eq.11)

Usage:
  python inference_acfg.py \\
      --prompt "A dog <c> in watercolor style <s>" \\
      --content_lora output/content/dog/pytorch_lora_weights.safetensors \\
      --style_lora output/style/watercolor/pytorch_lora_weights.safetensors \\
      --output_path results/acfg \\
      --guidance_omega 7.5
"""

import argparse
import copy
import logging
import os
import random

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

from craftlora_utils import (
    CONTENT_LAYERS, STYLE_LAYERS,
    filter_lora, scale_lora,
    detect_tokens, strip_tokens,
    aggregate_lora_weights,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timestep_activation(t: int, T: int, schedule_range: tuple) -> float:
    """
    Timestep-dependent activation (Eq.9 indicator functions).

    γ(t) = 1 if t ∈ T_schedule, else 0

    Args:
        t: Current timestep index (0 = final, T-1 = noisiest)
        T: Total number of timesteps
        schedule_range: (start_frac, end_frac) as fractions of T.
            The LoRA is active when start_frac * T <= t <= end_frac * T.

    NOTE: The paper does not fully specify T_c and T_s. This implementation
    provides a configurable interface. Default ranges should be tuned
    experimentally.
    """
    start = int(schedule_range[0] * T)
    end = int(schedule_range[1] * T)
    return 1.0 if start <= t <= end else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="CraftLoRA ACFG Inference (Algorithm 2)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--content_lora", type=str, default=None)
    parser.add_argument("--style_lora", type=str, default=None)
    parser.add_argument("--gamma_c", type=float, default=None,
                        help="Content strength override [0,1]. Auto-detected from <c> token if not set.")
    parser.add_argument("--gamma_s", type=float, default=None,
                        help="Style strength override [0,1]. Auto-detected from <s> token if not set.")
    parser.add_argument("--guidance_omega", type=float, default=7.5,
                        help="ACFG guidance strength ω (Eq.11)")
    parser.add_argument("--content_schedule", type=str, default="0.0,1.0",
                        help="Content LoRA active timestep range as 'start_frac,end_frac'")
    parser.add_argument("--style_schedule", type=str, default="0.0,1.0",
                        help="Style LoRA active timestep range as 'start_frac,end_frac'")
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--negative_prompt", type=str, default="low quality, bad quality, blurry")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    # Parse timestep schedules
    content_schedule = tuple(float(x) for x in args.content_schedule.split(","))
    style_schedule = tuple(float(x) for x in args.style_schedule.split(","))

    # Detect tokens from prompt
    tokens = detect_tokens(args.prompt)
    gamma_c = args.gamma_c if args.gamma_c is not None else tokens['gamma_c']
    gamma_s = args.gamma_s if args.gamma_s is not None else tokens['gamma_s']
    logger.info(f"Activation scalars: gamma_c={gamma_c}, gamma_s={gamma_s}")

    # Load models
    if args.vae_path and os.path.exists(args.vae_path):
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    model_id = args.model_path if args.model_path and os.path.exists(args.model_path) \
        else "stabilityai/stable-diffusion-xl-base-1.0"
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.float16
    ).to("cuda")

    # ACFG setup (Algorithm 2):
    # W_uncond = W_init (no LoRA) — the pipeline's UNet before loading LoRA
    # W_cond = W_init + aggregated LoRA updates
    logger.info("Creating W_uncond (backbone without LoRA) for ACFG...")
    unet_uncond = copy.deepcopy(pipeline.unet)

    # Load and aggregate LoRA weights (Eq.8)
    content_sd = {}
    if args.content_lora:
        raw_sd, _ = pipeline.lora_state_dict(args.content_lora)
        content_sd = raw_sd
        logger.info(f"Loaded content LoRA: {len(raw_sd)} keys")

    style_sd = {}
    if args.style_lora:
        raw_sd, _ = pipeline.lora_state_dict(args.style_lora)
        style_sd = raw_sd
        logger.info(f"Loaded style LoRA: {len(raw_sd)} keys")

    merged_lora = aggregate_lora_weights(content_sd, style_sd, gamma_c, gamma_s)
    pipeline.load_lora_into_unet(merged_lora, None, pipeline.unet)
    logger.info(f"Loaded {len(merged_lora)} aggregated LoRA keys into W_cond")

    # ACFG sampling (Algorithm 2)
    # ε_acfg = (1 + ω) * ε_cond - ω * ε_uncond   (Eq.11)
    #
    # NOTE: Full implementation of timestep-dependent LoRA activation (Eq.9)
    # requires modifying the denoising loop to switch LoRA weights per timestep.
    # The paper does not specify the exact T_c and T_s schedules.
    # Below we implement the core ACFG with fixed LoRA weights and provide
    # the timestep_activation function for future integration.

    logger.info(f"Generating with ACFG (ω={args.guidance_omega})...")
    safe_prompt = args.prompt[:30].replace(" ", "_")

    for img_idx in range(args.num_images_per_prompt):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + img_idx)

        # Get conditional prediction (with LoRA)
        with torch.no_grad():
            cond_output = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height, width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=0.0,
                generator=generator,
                output_type="latent",
            )
            cond_latents = cond_output.images

        # Get unconditional prediction (without LoRA, using W_init)
        original_unet = pipeline.unet
        pipeline.unet = unet_uncond
        with torch.no_grad():
            uncond_output = pipeline(
                prompt="",
                negative_prompt=None,
                height=args.height, width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=0.0,
                generator=torch.Generator(device="cuda").manual_seed(args.seed + img_idx),
                output_type="latent",
            )
            uncond_latents = uncond_output.images
        pipeline.unet = original_unet

        # ACFG combination (Eq.11)
        omega = args.guidance_omega
        acfg_latents = (1 + omega) * cond_latents - omega * uncond_latents

        # Decode
        with torch.no_grad():
            images = pipeline.vae.decode(acfg_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
            images = pipeline.image_processor.postprocess(images, output_type="pil")

        for i, img in enumerate(images):
            save_path = os.path.join(args.output_path, f"acfg_{img_idx}_{safe_prompt}.jpg")
            img.save(save_path)
            logger.info(f"Saved: {save_path}")

    torch.cuda.empty_cache()
    del unet_uncond
    logger.info(f"All ACFG results saved to: {args.output_path}")
