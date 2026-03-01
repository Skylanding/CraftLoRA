"""
CraftLoRA Rank-Limited Backbone Fine-Tuning (Contribution 1)

Implements the rank-constrained weight projection described in the paper:
  - Learnable basis matrices B_l per layer, QR-decomposed to orthogonal Q_l
  - Projection update: W_l = W_l^(0) - Q_l @ Q_l^T @ W_l^(0)   (Eq.1)
  - Hierarchical rank scheduling: r_l = r_max - (l-1)/(L-1) * (r_max - r_min)  (Eq.2)
  - Content/style subspace merging via concatenated QR   (Eq.3-4)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def compute_rank_schedule(num_layers: int, r_max: int, r_min: int) -> List[int]:
    """
    Hierarchical rank scheduling (Eq.2).

    r_l = r_max - (l-1)/(L-1) * (r_max - r_min),  l = 1,...,L

    Earlier layers get higher rank (more adaptation capacity for
    entangled low-level structure/texture), later layers get lower rank.

    Args:
        num_layers: Total number of layers L
        r_max: Maximum rank (for first layer)
        r_min: Minimum rank (for last layer)

    Returns:
        List of integer ranks, one per layer
    """
    if num_layers <= 1:
        return [r_max]
    ranks = []
    for l in range(num_layers):
        r_l = r_max - l / (num_layers - 1) * (r_max - r_min)
        ranks.append(max(1, round(r_l)))
    return ranks


class RankLimitedLayer(nn.Module):
    """
    Learnable basis matrix B_l for a single linear layer.

    The QR decomposition of B_l yields orthogonal basis Q_l, used to project
    out components from the frozen weight W_l^(0):

        W_l = W_l^(0) - Q_l @ Q_l^T @ W_l^(0)    (Eq.1)

    This constrains updates to directions orthogonal to the learned low-rank
    subspace, reducing overlap between content and style representations.
    """

    def __init__(self, weight_shape: Tuple[int, int], rank: int):
        super().__init__()
        d_out, d_in = weight_shape
        self.rank = min(rank, d_in, d_out)
        self.basis = nn.Parameter(torch.randn(d_in, self.rank) * 0.01)

    def get_orthogonal_basis(self) -> torch.Tensor:
        """QR decompose B_l to get orthogonal Q_l."""
        Q, _ = torch.linalg.qr(self.basis)
        return Q

    def project_weight(self, W0: torch.Tensor) -> torch.Tensor:
        """
        Apply rank-limited projection (Eq.1):
            W_l = W_l^(0) - Q_l @ Q_l^T @ W_l^(0)
        """
        Q = self.get_orthogonal_basis()
        return W0 - (Q @ Q.T @ W0.T).T


class RankLimitedBackbone(nn.Module):
    """
    Manages rank-limited fine-tuning for all target layers in a UNet.

    Supports separate content and style basis matrices (B_l^content, B_l^style)
    that are later merged via concatenated QR decomposition (Eq.3-4):

        Q_l^merged = QR([Q_l^content | Q_l^style])
        W_l^combined = W_l^(0) - Q_l^merged @ (Q_l^merged)^T @ W_l^(0)
    """

    def __init__(self, unet: nn.Module, target_modules: List[str],
                 rank_schedule: List[int]):
        """
        Args:
            unet: The UNet model (frozen weights W^(0))
            target_modules: List of module names to apply rank reduction
            rank_schedule: Per-layer rank values from compute_rank_schedule()
        """
        super().__init__()
        self.target_modules = target_modules

        self.content_bases = nn.ModuleDict()
        self.style_bases = nn.ModuleDict()

        for i, name in enumerate(target_modules):
            module = self._get_module(unet, name)
            if not hasattr(module, 'weight'):
                continue
            rank = rank_schedule[i] if i < len(rank_schedule) else rank_schedule[-1]
            safe_name = name.replace('.', '_')
            self.content_bases[safe_name] = RankLimitedLayer(module.weight.shape, rank)
            self.style_bases[safe_name] = RankLimitedLayer(module.weight.shape, rank)

    @staticmethod
    def _get_module(model: nn.Module, name: str) -> nn.Module:
        parts = name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    def compute_merged_weights(self, unet: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute rank-limited backbone W_init by merging content and style
        subspaces (Eq.3-4).

        For each layer:
            1. Get Q_l^content and Q_l^style
            2. Concatenate: [Q_l^content | Q_l^style]
            3. QR decompose the concatenation to get Q_l^merged
            4. W_l = W_l^(0) - Q_l^merged @ (Q_l^merged)^T @ W_l^(0)
        """
        merged = {}
        for name in self.target_modules:
            safe_name = name.replace('.', '_')
            if safe_name not in self.content_bases:
                continue

            module = self._get_module(unet, name)
            W0 = module.weight.data

            Q_content = self.content_bases[safe_name].get_orthogonal_basis()
            Q_style = self.style_bases[safe_name].get_orthogonal_basis()

            Q_concat = torch.cat([Q_content, Q_style], dim=1)
            Q_merged, _ = torch.linalg.qr(Q_concat)

            W_new = W0 - (Q_merged @ Q_merged.T @ W0.T).T
            merged[name] = W_new

        return merged

    def apply_to_unet(self, unet: nn.Module):
        """Apply merged rank-limited weights to the UNet, producing W_init."""
        merged_weights = self.compute_merged_weights(unet)
        with torch.no_grad():
            for name, W_new in merged_weights.items():
                module = self._get_module(unet, name)
                module.weight.copy_(W_new)

    def get_content_loss(self, unet: nn.Module, name: str) -> torch.Tensor:
        """
        Compute content-side projected weight for training B_l^content.
        Used during contrastive training with content supervision signal.
        """
        safe_name = name.replace('.', '_')
        module = self._get_module(unet, name)
        return self.content_bases[safe_name].project_weight(module.weight.data)

    def get_style_loss(self, unet: nn.Module, name: str) -> torch.Tensor:
        """
        Compute style-side projected weight for training B_l^style.
        Used during contrastive training with style supervision signal.
        """
        safe_name = name.replace('.', '_')
        module = self._get_module(unet, name)
        return self.style_bases[safe_name].project_weight(module.weight.data)
