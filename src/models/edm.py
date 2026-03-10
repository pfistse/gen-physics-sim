from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.base import BaseGenerativeModel
from metrics.distributed import (
    create_enstr_err_steps_metric,
    create_mean_std_err_steps_metric,
)
from metrics.rank0 import create_std_steps_metric


class EDMModel(BaseGenerativeModel):
    """Elucidated diffusion model."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        net: nn.Module,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 1.0,
        rho: float = 7.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        num_steps_eval: int = 18,
        use_heun: bool = True,
        eval_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.net = net

        self.save_hyperparameters(ignore=["net"])
        for k, v in {k: v for k, v in locals().items() if k != "self"}.items():
            setattr(self, k, v)

        self.register_distributed_metric(
            create_mean_std_err_steps_metric(unbiased=False)
        )
        self.register_distributed_metric(create_enstr_err_steps_metric(unbiased=False))
        self.register_rank0_metric(create_std_steps_metric())

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Skip-scale."""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Output-scale."""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input-scale."""
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Noise-scale."""
        return 0.25 * torch.log(sigma)

    def _sigma_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        step_indices = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (
            sigma_max_rho
            + step_indices / num_steps * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho
        sigmas[-1] = 0.0
        return sigmas

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor):
        """Denoise x at noise sigma with cond.

        x: [B, C, H, W]
        sigma: [B]
        cond: [B, S*C, H, W]
        """
        assert x.ndim == 4, f"x: [B,C,H,W], got {tuple(x.shape)}"
        assert sigma.ndim == 1, f"sigma: [B], got {tuple(sigma.shape)}"
        assert cond.ndim == 4, f"cond: [B,S*C,H,W], got {tuple(cond.shape)}"
        assert x.shape[0] == sigma.shape[0] == cond.shape[0], "batch size mismatch"

        c_in = self.c_in(sigma)
        c_out = self.c_out(sigma)
        c_skip = self.c_skip(sigma)

        x_in = torch.cat([cond, x * c_in[:, None, None, None]], dim=1)
        f_theta = self.net(x_in, self.c_noise(sigma))[:, -x.shape[1] :, :, :]
        return x * c_skip[:, None, None, None] + f_theta * c_out[:, None, None, None]

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Compute EDM loss.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert target.ndim == 5, f"target: [B,1,C,H,W], got {tuple(target.shape)}"
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        assert target.size(1) == 1, "target second dim == 1"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        x0 = target.squeeze(1)

        rnd = torch.randn(B, device=target.device)
        sigma = torch.exp(rnd * self.p_std + self.p_mean)
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)

        noise = torch.randn_like(x0)
        x = x0 + sigma[:, None, None, None] * noise

        d = self.denoise(x, sigma, cond_flat)

        weight = (sigma**2 + self.sigma_data**2) / (
            (sigma * self.sigma_data) ** 2
        )
        mse = (d - x0).square().mean(dim=(1, 2, 3))
        return (mse * weight).mean()

    @torch.no_grad()
    def generate_samples(
        self,
        cond: torch.Tensor,
        num_steps: int,
    ):
        """Sample next frame.

        cond: [B, S, C, H, W]
        """
        assert cond.ndim == 5, f"cond: [B,S,C,H,W], got {tuple(cond.shape)}"
        assert num_steps >= 1, "num_steps >= 1"

        device = cond.device
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        sigmas = self._sigma_schedule(num_steps, device)
        x = torch.randn(B, C, H, W, device=device) * sigmas[0]

        for i in range(num_steps):
            sigma = sigmas[i].expand(B)
            sigma_next = sigmas[i + 1].expand(B)

            denoised = self.denoise(x, sigma, cond_flat)
            d = (x - denoised) / sigma[:, None, None, None]
            x_next = x + (sigma_next - sigma)[:, None, None, None] * d

            if self.use_heun and sigma_next[0] > 0:
                denoised_next = self.denoise(x_next, sigma_next, cond_flat)
                d_next = (x_next - denoised_next) / sigma_next[:, None, None, None]
                x_next = x + (sigma_next - sigma)[:, None, None, None] * (
                    0.5 * d + 0.5 * d_next
                )

            x = x_next

        return x.unsqueeze(1)
