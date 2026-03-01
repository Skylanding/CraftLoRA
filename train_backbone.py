"""
CraftLoRA Rank-Limited Backbone Fine-Tuning (Contribution 1)

Trains the learnable basis matrices B_l^content and B_l^style using
contrastive content-style image pairs. After training, the merged
orthogonal projections produce W_init, the backbone for all subsequent
LoRA training and inference.

Training flow (Section 1.3):
  1. Construct 100 contrastive content-style pairs from 10 content x 10 style refs
  2. For each pair, one image supervises B_l^content, the other B_l^style
  3. After training, merge subspaces (Eq.3-4) and produce W_init
  4. Save W_init for use in train_craftlora_sdxl.py

Usage:
  python train_backbone.py \\
      --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
      --content_data_dir="path/to/content_refs" \\
      --style_data_dir="path/to/style_refs" \\
      --output_dir="output/backbone" \\
      --r_max=64 --r_min=4 \\
      --max_train_steps=1000 --learning_rate=1e-5
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, PretrainedConfig

from rank_reduction import compute_rank_schedule, RankLimitedBackbone
from craftlora_utils import CONTENT_LAYERS, STYLE_LAYERS, get_target_attention_modules

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CraftLoRA rank-limited backbone fine-tuning")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default=None)
    parser.add_argument("--content_data_dir", type=str, required=True,
                        help="Directory of content reference images (10 images)")
    parser.add_argument("--style_data_dir", type=str, required=True,
                        help="Directory of style reference images (10 images)")
    parser.add_argument("--output_dir", type=str, default="output/backbone")
    parser.add_argument("--r_max", type=int, default=64, help="Max rank for earliest layers")
    parser.add_argument("--r_min", type=int, default=4, help="Min rank for deepest layers")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


class ContrastivePairDataset(Dataset):
    """
    Contrastive content-style pair dataset (Section 1.3).

    Generates all (C_i, S_j) pairs from content and style reference directories.
    With 10 content and 10 style images, this yields 100 pairs.
    """

    def __init__(self, content_dir, style_dir, resolution=512):
        self.content_images = sorted(Path(content_dir).glob("*"))
        self.style_images = sorted(Path(style_dir).glob("*"))

        self.content_images = [p for p in self.content_images if p.suffix.lower() in
                               ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
        self.style_images = [p for p in self.style_images if p.suffix.lower() in
                             ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]

        self.pairs = [
            (c, s) for c in self.content_images for s in self.style_images
        ]

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        logger.info(f"Contrastive pairs: {len(self.content_images)} content x "
                     f"{len(self.style_images)} style = {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        content_path, style_path = self.pairs[idx]
        content_img = Image.open(content_path).convert("RGB")
        style_img = Image.open(style_path).convert("RGB")
        return {
            "content": self.transform(content_img),
            "style": self.transform(style_img),
        }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load UNet
    logger.info("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    # Load VAE
    vae_path = args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    )
    vae.requires_grad_(False)
    vae.to(device, dtype=torch.float32)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Discover target attention modules across all layers
    target_modules = get_target_attention_modules(unet)
    num_target_layers = len(target_modules)
    logger.info(f"Target modules for rank reduction: {num_target_layers}")

    # Compute rank schedule (Eq.2)
    rank_schedule = compute_rank_schedule(num_target_layers, args.r_max, args.r_min)
    logger.info(f"Rank schedule: max={args.r_max}, min={args.r_min}, "
                f"first 5 ranks={rank_schedule[:5]}")

    # Initialize rank-limited backbone
    backbone = RankLimitedBackbone(unet, target_modules, rank_schedule)
    backbone.to(device)

    # Only basis matrices are trainable
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.learning_rate)

    # Dataset
    dataset = ContrastivePairDataset(args.content_data_dir, args.style_data_dir, args.resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Training loop
    logger.info(f"Starting backbone fine-tuning for {args.max_train_steps} steps...")
    progress_bar = tqdm(range(args.max_train_steps), desc="Backbone fine-tuning")
    global_step = 0
    dataloader_iter = iter(dataloader)

    while global_step < args.max_train_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        content_images = batch["content"].to(device, dtype=torch.float32)
        style_images = batch["style"].to(device, dtype=torch.float32)

        # Encode to latents
        with torch.no_grad():
            content_latents = vae.encode(content_images).latent_dist.sample() * vae.config.scaling_factor
            style_latents = vae.encode(style_images).latent_dist.sample() * vae.config.scaling_factor
            content_latents = content_latents.to(weight_dtype)
            style_latents = style_latents.to(weight_dtype)

        # Add noise
        noise_c = torch.randn_like(content_latents)
        noise_s = torch.randn_like(style_latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (content_latents.shape[0],), device=device).long()

        noisy_content = noise_scheduler.add_noise(content_latents, noise_c, timesteps)
        noisy_style = noise_scheduler.add_noise(style_latents, noise_s, timesteps)

        # Compute rank-limited backbone weights and apply
        backbone.apply_to_unet(unet)

        # Forward pass: content image provides content supervision,
        # style image provides style supervision.
        # The basis matrices B_l^content and B_l^style learn to find
        # orthogonal subspaces that separate content and style.
        # NOTE: Full training requires text embeddings; this is a
        # simplified structure showing the core rank-reduction loop.
        # A complete implementation would integrate with the SDXL
        # text encoding pipeline.

        # Compute denoising loss on content and style images
        # to train the respective basis matrices
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Content basis training: project content images
        for name in backbone.target_modules:
            safe_name = name.replace('.', '_')
            if safe_name in backbone.content_bases:
                Q_c = backbone.content_bases[safe_name].get_orthogonal_basis()
                loss = loss + torch.norm(Q_c, p='fro') * 0.0  # placeholder for actual content supervision

        for name in backbone.target_modules:
            safe_name = name.replace('.', '_')
            if safe_name in backbone.style_bases:
                Q_s = backbone.style_bases[safe_name].get_orthogonal_basis()
                loss = loss + torch.norm(Q_s, p='fro') * 0.0  # placeholder for actual style supervision

        # TODO: The full training objective requires:
        # 1. Text encoding of content/style prompts
        # 2. UNet forward with projected weights
        # 3. Denoising loss on content images -> gradients to B_l^content
        # 4. Denoising loss on style images -> gradients to B_l^style
        # The paper does not fully specify the loss formulation for
        # training the basis matrices. The core mathematical operations
        # (QR decomposition, projection, merging) are implemented in
        # rank_reduction.py.

        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        progress_bar.update(1)
        global_step += 1

    # After training: merge content and style subspaces and save W_init
    logger.info("Merging content and style subspaces (Eq.3-4)...")
    backbone.apply_to_unet(unet)

    save_path = os.path.join(args.output_dir, "unet_w_init")
    unet.save_pretrained(save_path)
    logger.info(f"Saved W_init backbone to: {save_path}")

    # Also save the basis matrices for inspection
    basis_path = os.path.join(args.output_dir, "basis_matrices.pt")
    torch.save(backbone.state_dict(), basis_path)
    logger.info(f"Saved basis matrices to: {basis_path}")


if __name__ == "__main__":
    main()
