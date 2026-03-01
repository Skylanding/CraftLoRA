"""
CraftLoRA Utilities

Core utilities for CraftLoRA content-style disentanglement:
  - Disjoint layer sets I_c (content) and I_s (style) for the SDXL UNet
  - LoRA filtering and scaling per layer subset
  - Token detection for <c>/<s> routing signals

CraftLoRA's layer assignment extends the baseline two attention blocks
by adding low/mid/high layers to improve content-style coverage.
"""

from typing import List


# CraftLoRA disjoint layer assignment (Contribution 2, Eq.7):
#   I_c: content-sensitive layers (baseline content block + low/middle layers)
#   I_s: style-sensitive layers (baseline style block + high layers)
CONTENT_LAYERS = [
    'up_blocks.0.attentions.0',
    'down_blocks.0',
    'down_blocks.1',
    'mid_block',
]
STYLE_LAYERS = [
    'up_blocks.0.attentions.1',
    'up_blocks.1',
    'up_blocks.2',
    'up_blocks.3',
]


def is_belong_to_blocks(key: str, blocks: List[str]) -> bool:
    """Check whether a parameter key belongs to any of the specified blocks."""
    for block in blocks:
        if block in key:
            return True
    return False


def filter_lora(state_dict: dict, blocks: List[str]) -> dict:
    """Filter LoRA state dict to only include keys belonging to specified blocks."""
    return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks)}


def scale_lora(state_dict: dict, alpha: float) -> dict:
    """Scale all LoRA weights by a constant factor (gamma_c or gamma_s)."""
    return {k: v * alpha for k, v in state_dict.items()}


def detect_tokens(prompt: str) -> dict:
    """
    Detect <c> and <s> routing tokens in a prompt (Contribution 2).

    Returns activation scalars gamma_c and gamma_s:
      - Default: {0, 1} based on token presence
      - Users can override with continuous values in [0, 1]
    """
    return {
        'gamma_c': 1.0 if '<c>' in prompt else 0.0,
        'gamma_s': 1.0 if '<s>' in prompt else 0.0,
    }


def strip_tokens(prompt: str) -> str:
    """
    Remove <c>/<s> tokens from prompt to get clean text for encoding (Eq.6).

    e_sem = SDXL_Encoder(p \\ {<c>, </c>, <s>, </s>})
    """
    for token in ['<c>', '</c>', '<s>', '</s>']:
        prompt = prompt.replace(token, '')
    return ' '.join(prompt.split())


def get_target_attention_modules(unet, blocks: List[str] = None) -> List[str]:
    """
    Discover attention modules for LoRA injection, optionally filtered by block.

    If blocks is None, returns all attention modules.
    """
    if blocks is None:
        blocks = CONTENT_LAYERS + STYLE_LAYERS

    attns = [
        attn_name.rsplit('.', 1)[0]
        for attn_name, _ in unet.attn_processors.items()
        if is_belong_to_blocks(attn_name, blocks)
    ]

    target_modules = [
        f'{attn}.{mat}'
        for mat in ['to_k', 'to_q', 'to_v', 'to_out.0']
        for attn in attns
    ]
    return target_modules


def aggregate_lora_weights(
    content_state_dict: dict,
    style_state_dict: dict,
    gamma_c: float = 1.0,
    gamma_s: float = 1.0,
) -> dict:
    """
    Aggregate content and style LoRA weights (Eq.8).

    W_agg = W_init + sum_{i in I_c} E_i(gamma_c * dW_i^c) + sum_{i in I_s} E_i(gamma_s * dW_i^s)

    This filters each state dict to its designated layer set, scales by the
    activation scalar, and merges them. The caller is responsible for loading
    the result into the UNet on top of W_init.

    Args:
        content_state_dict: Full LoRA state dict from content training
        style_state_dict: Full LoRA state dict from style training
        gamma_c: Content activation scalar (0 or 1, or continuous in [0,1])
        gamma_s: Style activation scalar (0 or 1, or continuous in [0,1])
    """
    content_lora = filter_lora(content_state_dict, CONTENT_LAYERS)
    content_lora = scale_lora(content_lora, gamma_c)

    style_lora = filter_lora(style_state_dict, STYLE_LAYERS)
    style_lora = scale_lora(style_lora, gamma_s)

    return {**content_lora, **style_lora}
