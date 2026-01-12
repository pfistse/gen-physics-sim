import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import logging

from models.base import BaseGenerativeModel

logger = logging.getLogger(__name__)


class UNet(BaseGenerativeModel):
    """Simple U-Net baseline for physics simulations."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        net: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        num_steps_eval: int = 1,
        eval_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])

        self.dim = dim
        self.sim_fields = sim_fields
        self.sim_params = sim_params
        self.ctx_len = ctx_len
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps_eval = num_steps_eval
        self.eval_config = eval_config

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Return the training loss for a batch.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert target.ndim == 5, f"target: [B,1,C,H,W], got {target.shape}"
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {cond.shape}"
        assert target.size(1) == 1, "target.size(1) must be 1"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        target_frame = target.squeeze(1)

        # Dummy timestep (0.0)
        t = torch.zeros((B,), device=cond.device, dtype=cond.dtype)

        noise = torch.randn_like(target_frame)
        x_in = torch.cat([cond_flat, noise], dim=1)

        pred = self.net(x_in, t)
        loss = F.mse_loss(pred, target_frame)
        return loss

    def generate_samples(self, cond: torch.Tensor, num_steps: int = 1, **kwargs):
        """Sample the next frame from ``cond``.

        cond: [B, S, C, H, W]
        return: [B, 1, C, H, W]
        """
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {cond.shape}"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        # Dummy timestep (0.0)
        t = torch.zeros((B,), device=cond.device, dtype=cond.dtype)

        noise = torch.randn((B, C, H, W), device=cond.device)
        x_in = torch.cat([cond_flat, noise], dim=1)

        pred = self.net(x_in, t)
        return pred.unsqueeze(1)
