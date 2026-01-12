import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import logging

from models.base import BaseGenerativeModel
from metrics.distributed import create_mean_std_rel_err_steps_metric
from metrics.rank0 import create_std_steps_metric

logger = logging.getLogger(__name__)


class FlowMatchingModel(BaseGenerativeModel):
    """Flow matching model."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        net: nn.Module,
        sigma_min: float = 0.001,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        num_steps_eval: int = 1,
        integrator: str = "euler",
        eval_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"])
        for k, v in {k: v for k, v in locals().items() if k != "self"}.items():
            setattr(self, k, v)

        self._integrate = {
            "euler": self._integrate_euler,
            "rk4": self._integrate_rk4,
        }[self.integrator]

        self.register_distributed_metric(
            create_mean_std_rel_err_steps_metric(unbiased=False)
        )
        self.register_rank0_metric(create_std_steps_metric())

    def phi_t(self, x_0, x_1, t):
        """Interpolate x0→x1 at t.

        x_0: [B, C, H, W]
        x_1: [B, C, H, W]
        t: [B]
        """
        t = t.view(-1, 1, 1, 1)
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def v_t(self, x_0, x_1, t):
        """Vector field at t.

        x_0: [B, C, H, W]
        x_1: [B, C, H, W]
        t: [B]
        """
        return x_1 - (1 - self.sigma_min) * x_0

    @staticmethod
    def _integrate_euler(f, x_0, t_0, t_1, dt):
        """Euler ODE solver."""
        t_0 = torch.as_tensor(t_0, dtype=x_0.dtype, device=x_0.device)
        t_1 = torch.as_tensor(t_1, dtype=x_0.dtype, device=x_0.device)
        dt = torch.as_tensor(dt, dtype=x_0.dtype, device=x_0.device)

        with torch.no_grad():
            t = t_0
            x = x_0
            while t < t_1:
                dt = torch.min(abs(dt), abs(t_1 - t))
                x, t = x + dt * f(t, x), t + dt

        return x

    @staticmethod
    def _integrate_rk4(f, x_0, t_0, t_1, dt):
        """RK4 ODE solver."""
        t_0 = torch.as_tensor(t_0, dtype=x_0.dtype, device=x_0.device)
        t_1 = torch.as_tensor(t_1, dtype=x_0.dtype, device=x_0.device)
        dt = torch.as_tensor(dt, dtype=x_0.dtype, device=x_0.device)

        with torch.no_grad():
            t = t_0.clone()
            x = x_0.clone()
            while t < t_1:
                step = torch.min(dt, t_1 - t)
                k1 = step * f(t, x)
                k2 = step * f(t + step / 2, x + k1 / 2)
                k3 = step * f(t + step / 2, x + k2 / 2)
                k4 = step * f(t + step, x + k3)
                x.add_((k1 + 2 * k2 + 2 * k3 + k4) / 6)
                t.add_(step)

        return x

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor, debug: bool = False):
        """Compute training loss.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert target.ndim == 5, f"target: [B,1,C,H,W], got {tuple(target.shape)}"
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        assert target.size(1) == 1, "target second dim == 1"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        x_1 = target.squeeze(1)

        x_0 = torch.randn_like(x_1)
        t = torch.rand((B,), device=target.device)

        x_t = self.phi_t(x_0, x_1, t)
        v_t = self.v_t(x_0, x_1, t)

        x_in = torch.cat([cond_flat, x_t], dim=1)
        v_pred = self.net(x_in, t)

        loss = F.mse_loss(v_pred, v_t)

        if debug:
            return loss, v_pred.unsqueeze(1), v_t.unsqueeze(1), None, cond_flat.unsqueeze(1)
        return loss

    def generate_samples(self, cond: torch.Tensor, num_steps: int):
        """Sample next frame.

        cond: [B, S, C, H, W]
        """
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        x = torch.randn(B, C, H, W, device=cond.device)
        dt = 1.0 / num_steps

        with torch.no_grad():

            def wrapper(t_scalar: float, x_t: torch.Tensor):
                t_vec = torch.full((x_t.shape[0],), float(t_scalar), dtype=x_t.dtype, device=x_t.device)
                x_in = torch.cat([cond_flat, x_t], dim=1)
                v_pred = self.net(x_in, t_vec)
                return v_pred

            x_1 = self._integrate(wrapper, x, 0.0, 1.0, dt)
            return x_1.unsqueeze(1)
