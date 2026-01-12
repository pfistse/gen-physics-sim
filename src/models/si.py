from typing import Dict, List, Optional

import torch
from torch import nn

from models.base import BaseGenerativeModel
from metrics.distributed import (
    create_mean_std_rel_err_steps_metric,
    create_std_rel_err_grid_metric,
)


class Interpolant:
    """Sigma-scaled interpolant as defined in the Foellmer baseline."""

    def __init__(self, sigma_coef: float, beta_fn: str, foellmer_process: bool) -> None:
        assert beta_fn in {"t", "t^2"}, "beta_fn must be 't' or 't^2'"
        self.sigma_coef = sigma_coef
        self.beta_fn = beta_fn
        self.foellmer_process = foellmer_process

    @staticmethod
    def _expand(t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 1, "t: [B]"
        return t[:, None, None, None]

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        if self.beta_fn == "t^2":
            return t.pow(2)
        return t

    def _beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        if self.beta_fn == "t^2":
            return 2.0 * t
        return torch.ones_like(t)

    def _sigma_scalar(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_coef * (1.0 - t)

    def _g_F_scalar(self, t: torch.Tensor) -> torch.Tensor:
        assert self.beta_fn == "t^2"
        return self.sigma_coef * torch.sqrt((3.0 - t) * (1.0 - t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._expand(self._sigma_scalar(t))

    def g_s(self, t: torch.Tensor, g_coef: Optional[float] = None) -> torch.Tensor:
        """Return valid diffusion coefficient g_s(t).

        g_coef scales the *excess variance* g_F^2 - σ^2:
            g^2 = σ^2 + g_coef * (g_F^2 - σ^2)
        """
        sigma = self._sigma_scalar(t)

        if not self.foellmer_process:
            return self._expand(sigma)

        g_F = self._g_F_scalar(t)
        lam = 1.0 if g_coef is None else g_coef

        g2 = sigma**2 + lam * (g_F**2 - sigma**2)
        g2 = g2.clamp_min(0.0)
        g = g2.sqrt()
        return self._expand(g)

    def evaluate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return interpolant dictionary for given batch."""
        assert x0.shape == x1.shape == noise.shape, "z0, z1, noise shapes must match"

        at, bt = 1.0 - t, self._beta(t)
        adot, bdot = -torch.ones_like(t), self._beta_dot(t)
        st = self.sigma_coef * at
        sdot = -self.sigma_coef * torch.ones_like(t)
        root_t = torch.sqrt(t.clamp_min(0.0))

        at_w, bt_w, adot_w, bdot_w, st_w, sdot_w, root_t_w = map(
            self._expand, (at, bt, adot, bdot, st, sdot, root_t)
        )

        gamma_t = st_w * root_t_w

        xt = at_w * x0 + bt_w * x1 + gamma_t * noise
        drift_target = adot_w * x0 + bdot_w * x1 + (sdot_w * root_t_w) * noise

        return xt, drift_target


class StochasticInterpolation(BaseGenerativeModel):
    """Stochastic interpolation baseline."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        net: nn.Module,
        sigma_coef: float = 1.0,
        beta_fn: str = "t^2",
        t_min_train: float = 0.0,
        t_max_train: float = 0.999,
        t_min_sampling: float = 0.0,
        t_max_sampling: float = 0.999,
        foellmer_process: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        num_steps_eval: Optional[int] = None,
        eval_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["net"])
        for k, v in {k: v for k, v in locals().items() if k != "self"}.items():
            setattr(self, k, v)

        self.interpolant = Interpolant(
            sigma_coef=self.sigma_coef,
            beta_fn=self.beta_fn,
            foellmer_process=self.foellmer_process,
        )

        self.register_distributed_metric(
            create_mean_std_rel_err_steps_metric(unbiased=True)
        )
        self.register_distributed_metric(create_std_rel_err_grid_metric(unbiased=True))

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Compute Foellmer drift loss.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert target.ndim == 5, f"target: [B,1,C,H,W], got {tuple(target.shape)}"
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        assert target.size(1) == 1, "target second dim == 1"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        x0 = cond[:, -1]
        x1 = target.squeeze(1)

        t = (
            torch.rand(B, device=target.device) * (self.t_max_train - self.t_min_train)
            + self.t_min_train
        )
        noise = torch.randn_like(x0)

        xt, drift_target = self.interpolant.evaluate(x0=x0, x1=x1, t=t, noise=noise)
        drift_pred = self.net(torch.cat([cond_flat, xt], dim=1), t)

        loss = (drift_pred - drift_target).square().sum(dim=(1, 2, 3)).mean()
        return loss

    @torch.no_grad()
    def generate_samples(
        self,
        cond: torch.Tensor,
        num_steps: int,
        g_coef: Optional[float] = None,
    ):
        """Sample next frame.

        cond: [B, S, C, H, W]
        """
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        ts = torch.linspace(
            self.t_min_sampling,
            self.t_max_sampling,
            num_steps + 1,
            device=cond.device,
            dtype=cond.dtype,
        )

        x0 = cond[:, -1]
        xt = x0.clone()

        for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:])):
            dt = (t1 - t0).to(cond.dtype)
            t = torch.full((B,), float(t0), device=cond.device, dtype=cond.dtype)

            drift_ref = self.net(torch.cat([cond_flat, xt], dim=1), t)
            sigma_t = self.interpolant.sigma(t).to(xt.dtype)  # [B,1,1,1]
            w = torch.randn_like(xt)

            if self.foellmer_process and i > 0:
                beta = self.interpolant._beta(t)
                beta_dot = self.interpolant._beta_dot(t)
                alpha = 1.0 - t
                alpha_dot = -torch.ones_like(t)

                sigma = self.sigma_coef * (1.0 - t)
                sigma_dot = -self.sigma_coef * torch.ones_like(t)

                A = 1.0 / (t * sigma * (beta_dot * sigma - beta * sigma_dot))
                coef_x0 = beta * alpha_dot - beta_dot * alpha

                beta_w, beta_dot_w, A_w, coef_x0_w = map(
                    self.interpolant._expand, (beta, beta_dot, A, coef_x0)
                )

                c = beta_dot_w * xt + coef_x0_w * x0

                score = A_w * (beta_w * drift_ref - c)

                g_t = self.interpolant.g_s(t, g_coef=g_coef).to(xt.dtype)
                drift = drift_ref + 0.5 * (g_t**2 - sigma_t**2) * score

                xt = xt + drift * dt + g_t * w * dt.sqrt()
            else:
                xt = xt + drift_ref * dt + sigma_t * w * dt.sqrt()

        return xt.unsqueeze(1)


__all__ = ["StochasticInterpolation"]
