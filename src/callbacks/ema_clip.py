from __future__ import annotations

from typing import Iterable, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer


def _clip_grad_norm(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    clip_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tuple[torch.Tensor, bool]:
    """Clip gradients if their total norm exceeds ``max_norm`` by scaling to ``clip_norm``.

    Returns (total_norm, clipped_flag).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p is not None and p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0), False

    device = grads[0].device
    norm_type = float(norm_type)

    if norm_type == float("inf"):
        total_norm = max(g.detach().abs().max().to(device) for g in grads)
    else:
        norms = [torch.linalg.vector_norm(g.detach(), ord=norm_type) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack([n.to(device) for n in norms]), ord=norm_type)

    if error_if_nonfinite and (torch.isnan(total_norm) or torch.isinf(total_norm)):
        raise RuntimeError(
            "Total gradient norm is non-finite. Set error_if_nonfinite=False to ignore."
        )

    max_norm = float(max_norm)
    clip_norm = float(clip_norm)
    clipped = bool(total_norm > max_norm)

    if clipped:
        scale = clip_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale.to(g.device))

    return total_norm, clipped


class EmaGradClip(Callback):
    """EMA-based adaptive gradient clipping.

    Tracks two EMAs of gradient norms with different coefficients and:
    - Triggers clipping when current total norm > ``max_norm_ratio * ema2``
    - If triggered, scales gradients to ``clip_norm_ratio * ema1``

    This mirrors pde-transformer’s EmaGradClip.
    """

    def __init__(
        self,
        ema_coef1: float = 0.9,
        ema_coef2: float = 0.99,
        max_norm_ratio: float = 2.0,
        clip_norm_ratio: float = 1.1,
    ) -> None:
        super().__init__()
        self.ema_coef1 = float(ema_coef1)
        self.ema_coef2 = float(ema_coef2)
        self.max_norm_ratio = float(max_norm_ratio)
        self.clip_norm_ratio = float(clip_norm_ratio)
        self._ema1 = 0.0
        self._ema2 = 0.0
        self._t = 0

    def _record(self, new_norm: float) -> None:
        self._t += 1
        self._ema1 = self.ema_coef1 * self._ema1 + (1.0 - self.ema_coef1) * float(new_norm)
        self._ema2 = self.ema_coef2 * self._ema2 + (1.0 - self.ema_coef2) * float(new_norm)

    @property
    def current_ema1(self) -> float:
        if self._t == 0:
            return 0.0
        return self._ema1 / (1.0 - self.ema_coef1**self._t)

    @property
    def current_ema2(self) -> float:
        if self._t == 0:
            return 0.0
        return self._ema2 / (1.0 - self.ema_coef2**self._t)

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Optimizer,
    ) -> None:
        # Bootstrap: no EMA yet → compute norm and record without clipping (huge thresholds)
        if self._ema2 == 0.0:
            total_norm, clipped = _clip_grad_norm(
                pl_module.parameters(), max_norm=1e12, clip_norm=1.0
            )
        else:
            max_norm = self.max_norm_ratio * self.current_ema2
            clip_norm = self.clip_norm_ratio * self.current_ema1
            total_norm, clipped = _clip_grad_norm(
                pl_module.parameters(), max_norm=max_norm, clip_norm=clip_norm
            )

        # If clipped, record the effective clip_norm; otherwise record the actual total norm
        norm_to_record = (
            self.clip_norm_ratio * self.current_ema1 if clipped and self._t != 0 else float(total_norm)
        )
        self._record(norm_to_record)
