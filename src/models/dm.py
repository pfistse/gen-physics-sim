import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
import logging
from typing import List, Optional, Dict

from utils.wandb import fig_to_image

from models.components.unet import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)


from models.base import BaseGenerativeModel
from metrics.distributed import create_mean_std_rel_err_steps_metric
from metrics.rank0 import create_denoising_metric

logger = logging.getLogger(__name__)


class DiffusionModel(BaseGenerativeModel):
    """Denoising diffusion model for physics simulations."""

    def __init__(
        self,
        dim: int,
        sim_fields: List[str],
        sim_params: List[str],
        ctx_len: int,
        net: nn.Module,
        timesteps: int = 50,
        beta_schedule: str = "linear",
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        ddim_sampling: bool = False,
        num_steps_eval: int = 50,
        eval_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["net"])
        for k, v in {k: v for k, v in locals().items() if k != "self"}.items():
            setattr(self, k, v)

        schedule_fns = {
            "linear": linear_beta_schedule,
            "cosine": cosine_beta_schedule,
            "quadratic": quadratic_beta_schedule,
            "sigmoid": sigmoid_beta_schedule,
        }
        betas = schedule_fns[self.beta_schedule](self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        alphas_cumprod_prev = torch.cat(
            [
                torch.ones(1, device=betas.device, dtype=betas.dtype),
                alphas_cumprod[:-1],
            ],
            dim=0,
        )

        posterior_var = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).clamp_min(1e-20)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )
        self.register_buffer("posterior_var", posterior_var)
        self.register_rank0_metric(create_denoising_metric())
        self.register_distributed_metric(
            create_mean_std_rel_err_steps_metric(unbiased=False)
        )

    def compute_loss(
        self, target: torch.Tensor, cond: torch.Tensor, debug: bool = False
    ):
        """Return the training loss for a batch.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert len(target.shape) == 5, f"Expected [B, 1, C, H, W], got {target.shape}"
        assert len(cond.shape) == 5, f"Expected [B, S, C, H, W], got {cond.shape}"
        assert target.size(1) == 1, "Second dimension of `target` must be 1"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        target_frame = target.squeeze(1)

        t_idx = torch.randint(0, self.timesteps, (B,), device=target.device)
        t = t_idx.float() / (self.timesteps - 1)

        noise_target = torch.randn_like(target_frame)
        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].view(B, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx].view(B, 1, 1, 1)

        x_t = sqrt_alpha * target_frame + sqrt_one_minus * noise_target

        x_t_cond = cond_flat

        model_in = torch.cat([x_t_cond, x_t], dim=1)
        eps_pred = self.net(model_in, t)
        loss = F.mse_loss(eps_pred, noise_target)

        if debug:
            return loss, eps_pred, noise_target
        return loss

    def generate_samples(self, cond: torch.Tensor, num_steps: int):
        """Sample the next frame from ``cond``.

        cond: [B, S, C, H, W]
        return: [B, 1, C, H, W]
        """
        # eta=0 corresponds to DDIM, eta=1 corresponds to DDPM
        eta = 0.0 if self.ddim_sampling else 1.0

        if num_steps != self.timesteps:
            assert self.ddim_sampling, "num_steps!=timesteps requires ddim_sampling"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        x = torch.randn(B, C, H, W, device=cond.device)

        t_range = torch.linspace(
            self.timesteps - 1, 0, num_steps, dtype=torch.long, device=cond.device
        )

        for i, t_idx in enumerate(t_range):
            t_idx = int(t_idx.item())
            t = torch.full((B,), t_idx / (self.timesteps - 1), device=cond.device)

            model_in = torch.cat([cond_flat, x], dim=1)
            eps_pred = self.net(model_in, t)

            alpha_bar = self.alphas_cumprod[t_idx]

            if i < len(t_range) - 1:
                prev_t_idx = int(t_range[i + 1].item())
                alpha_bar_prev = self.alphas_cumprod[prev_t_idx]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=cond.device)

            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            )

            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred

            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

        samples = x.unsqueeze(1)
        return samples

    def on_train_start(self):
        """Log noise schedules to W&B at the start of training."""
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.arange(self.timesteps)

        ax.plot(steps, self.betas.cpu().numpy(), label="betas")
        ax.plot(steps, self.alphas_cumprod.cpu().numpy(), label="alphas_cumprod")
        ax.plot(
            steps, self.sqrt_alphas_cumprod.cpu().numpy(), label="sqrt_alphas_cumprod"
        )
        ax.plot(
            steps,
            self.sqrt_one_minus_alphas_cumprod.cpu().numpy(),
            label="sqrt_one_minus_alphas_cumprod",
        )
        ax.plot(steps, self.posterior_var.cpu().numpy(), label="posterior_var")

        ax.set_title(f"Noise Schedule: {self.beta_schedule}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        self.logger.experiment.log(
            {"model/noise_schedule": wandb.Image(fig_to_image(fig))}
        )
        plt.close(fig)
