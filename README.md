<h3 align="center">
<b>CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion</b>
<br>
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2602.18936">
    <img src="https://img.shields.io/badge/arXiv-2602.18936-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/Skylanding/CraftLoRA">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
</p>


![Teaser](figure/teaser.pdf)

## Abstract

Personalized image generation requires effectively balancing content fidelity with stylistic consistency when synthesizing images based on text and reference examples. Low-Rank Adaptation (LoRA) offers an efficient personalization approach, with potential for precise control through combining LoRA weights on different concepts. However, existing combination techniques face persistent challenges: entanglement between content and style representations, insufficient guidance for controlling elements' influence, and unstable weight fusion that often require additional training. We address these limitations through CRAFT-LoRA, with complementary components: (1) rank-constrained backbone fine-tuning that injects low-rank projection residuals to encourage learning decoupled content and style subspaces; (2) a prompt-guided approach featuring an expert encoder with specialized branches that enables semantic extension and precise control through selective adapter aggregation; and (3) a training-free, timestep-dependent classifier-free guidance scheme that enhances generation stability by strategically adjusting noise predictions across diffusion steps. Our method significantly improves content-style disentanglement, enables flexible semantic control over LoRA module combinations, and achieves high-fidelity generation without additional retraining overhead.

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Stage 1: Rank-Constrained Backbone Fine-Tuning

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
@article{li2026craft,
  title={CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion},
  author={Li, Yu and Cai, Yujun and Zhang, Chi},
  journal={arXiv preprint arXiv:2602.18936},
  year={2026}
}
```


## License

This project is licensed under the [MIT License](LICENSE).
