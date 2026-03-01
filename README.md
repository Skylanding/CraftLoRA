<h3 align="center">
<b>CraftLoRA: Crafting Style-Content Disentangled LoRA for Diffusion Models</b>
<br>
</h3>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
</p>

![Teaser](docs/teaser_craftlora.png)

## Abstract

<!-- TODO: paste abstract here -->

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Stage 1: Rank-Limited Backbone Fine-Tuning

```bash
python train_backbone.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --content_data_dir="path/to/content_refs" \
    --style_data_dir="path/to/style_refs" \
    --output_dir="output/backbone" \
    --r_max=64 --r_min=4 \
    --max_train_steps=1000 --learning_rate=1e-5
```

### Stage 2: Content / Style LoRA Training

```bash
# Content LoRA
accelerate launch train_craftlora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_backbone_path="output/backbone/unet_w_init" \
    --instance_data_dir="path/to/content_images" \
    --output_dir="output/content/subject_name" \
    --instance_prompt="A subject_name <c>" \
    --resolution=512 --rank=64 \
    --train_batch_size=1 --learning_rate=1e-5 \
    --max_train_steps=1000 --seed=0 \
    --gradient_checkpointing --use_8bit_adam --mixed_precision="fp16"

# Style LoRA
accelerate launch train_craftlora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_backbone_path="output/backbone/unet_w_init" \
    --instance_data_dir="path/to/style_images" \
    --output_dir="output/style/style_name" \
    --instance_prompt="In style_name style <s>" \
    --resolution=512 --rank=64 \
    --train_batch_size=1 --learning_rate=1e-5 \
    --max_train_steps=1000 --seed=0 \
    --gradient_checkpointing --use_8bit_adam --mixed_precision="fp16"
```

## Inference

### Standard Inference

```bash
python inference.py \
    --prompt="A dog <c> in watercolor style <s>" \
    --content_lora="output/content/dog/pytorch_lora_weights.safetensors" \
    --style_lora="output/style/watercolor/pytorch_lora_weights.safetensors" \
    --output_path="results/combined"
```

### ACFG Inference

```bash
python inference_acfg.py \
    --prompt="A cat <c> in oil painting style <s>" \
    --content_lora="output/content/cat/pytorch_lora_weights.safetensors" \
    --style_lora="output/style/oil_painting/pytorch_lora_weights.safetensors" \
    --output_path="results/acfg" \
    --guidance_omega=7.5
```

## Citation

```bibtex
@misc{craftlora2024,
    title={CraftLoRA: Crafting Style-Content Disentangled LoRA for Diffusion Models},
    year={2024},
}
```

## License

This project is licensed under the [MIT License](LICENSE).
