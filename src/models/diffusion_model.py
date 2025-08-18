import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import List, Optional, Dict
from dataclasses import dataclass, field, fields

from utils.log import get_logger
from utils.metrics import calculate_mse, calculate_vrs
from utils.wandb import (
    log_samples_video,
    log_comparison_video,
    log_mse,
    fig_to_image,
    log_denoising_video,
)
from models.components.unet import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)


logger = get_logger("models.dm")


@dataclass
class DiffusionModelCfg:
    dim: int
    sim_fields: List[str]
    sim_params: List[str]
    ctx_len: int
    net: nn.Module
    timesteps: int = 50
    beta_schedule: str = "linear"
    cond_noising: bool = True
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_steps_eval: int = 50
    eval_config: Dict = field(default_factory=dict)


class DiffusionModel(pl.LightningModule):
    """Denoising diffusion model for physics simulations."""

    def __init__(self, cfg: DiffusionModelCfg):
        super().__init__()

        self.net = cfg.net
        for f in fields(cfg):
            if f.name != "net":
                setattr(self, f.name, getattr(cfg, f.name))

        self.save_hyperparameters(
            {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "net"}
        )

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
            [torch.ones(1, device=betas.device, dtype=betas.dtype),
             alphas_cumprod[:-1]],
            dim=0
        )

        posterior_var = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).clamp_min(1e-20)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("posterior_var", posterior_var)

    def diffusion_loss(self, target: torch.Tensor, cond: torch.Tensor, debug: bool = False):
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
        noise_cond = torch.randn_like(cond_flat)

        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].view(B, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx].view(B, 1, 1, 1)

        x_t = sqrt_alpha * target_frame + sqrt_one_minus * noise_target

        if self.cond_noising:
            x_t_cond = sqrt_alpha * cond_flat + sqrt_one_minus * noise_cond
        else:
            x_t_cond = cond_flat

        model_in = torch.cat([x_t_cond, x_t], dim=1)
        eps_pred = self.net(model_in, t)
        loss = F.mse_loss(eps_pred, noise_target)

        if debug:
            return loss, eps_pred, noise_target
        return loss

    def generate_samples(self, cond: torch.Tensor, num_steps: int = None, debug: bool = False):
        """Sample the next frame from ``cond``.

        cond: [B, S, C, H, W]
        return: [B, 1, C, H, W]
        """
        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        x = torch.randn(B, C, H, W, device=cond.device)

        if debug:
            denoising_steps = [x.cpu().clone()]

        steps = num_steps if num_steps is not None else self.timesteps
        t_range = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long)
        for t_idx in t_range:
            t_idx = int(t_idx.item())
            t = torch.full((B,), t_idx / (self.timesteps - 1), device=cond.device)
            if self.cond_noising:
                sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].view(1, 1, 1, 1)
                sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_idx].view(1, 1, 1, 1)
                noise_cond = torch.randn_like(cond_flat)
                x_t_cond = sqrt_alpha * cond_flat + sqrt_one_minus * noise_cond
            else:
                x_t_cond = cond_flat

            model_in = torch.cat([x_t_cond, x], dim=1)
            eps_pred = self.net(model_in, t)

            beta = self.betas[t_idx]
            alpha_bar = self.alphas_cumprod[t_idx]
            x_mean = (x - beta / torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(1 - beta)
            if t_idx > 0:
                x = x_mean + torch.sqrt(self.posterior_var[t_idx]) * torch.randn_like(x)
            else:
                x = x_mean

            if debug:
                denoising_steps.append(x.cpu().clone())

        samples = x.unsqueeze(1)
        if debug:
            denoising_steps = torch.stack(denoising_steps, dim=0)  # [steps+1, B, C, H, W]
            return samples, denoising_steps
        return samples

    def generate_sequence(self, cond: torch.Tensor, seq_len: int, num_steps: int = None):
        """Generate ``seq_len`` sequential samples.

        cond: [B, S, C, H, W]
        return: [B, seq_len, C, H, W]
        """
        B, S, C, H, W = cond.shape
        P = len(self.sim_params)
        F = C - P
        cond_len = S
        cond_window = cond.clone()
        gen_seq = []
        const_params = None
        if P > 0:
            const_params = cond[:, :1, -P:, :, :]
        for _ in range(seq_len):
            next_frame = self.generate_samples(cond_window, num_steps=num_steps)
            if const_params is not None:
                next_frame[:, :, -P:, :, :] = const_params
            gen_seq.append(next_frame)
            if cond_len > 1:
                cond_window = torch.cat([cond_window[:, 1:], next_frame], dim=1)
            else:
                cond_window = next_frame
        return torch.cat(gen_seq, dim=1)

    def prediction_step(self, cond: torch.Tensor, num_steps: int):
        """Prediction step for inference mode."""
        return self.generate_samples(cond, num_steps=num_steps)

    def forward(
        self,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: int = None,
    ):
        """Training and inference entry point.

        cond: [B, S, C, H, W]
        target: optional [B, 1, C, H, W]
        """
        if target is not None:
            return self.diffusion_loss(target, cond)
        return self.prediction_step(cond, num_steps if num_steps is not None else self.timesteps)

    def training_step(self, batch, batch_idx):
        """Training step for Lightning module."""
        cond, target = batch
        loss = self.forward(cond, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning module."""
        self.test_step(batch, batch_idx)
        self.log("val_loss", 0, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """Test step for Lightning module."""
        if batch_idx == 0:
            # MSE metric
            mse_cfg = self.eval_config.metrics.mse
            if mse_cfg.enabled:
                mse_metrics = calculate_mse(
                    model=self,
                    channel_titles=self.eval_config.channel_titles,
                    num_steps=self.num_steps_eval,
                    **mse_cfg,
                )
                log_mse(mse_metrics, self, prefix="test")

            # VRS metric
            vrs_cfg = self.eval_config.metrics.vrs
            if vrs_cfg.enabled:
                vrs_metrics = calculate_vrs(model=self, **vrs_cfg)
                for k, v in vrs_metrics.items():
                    self.log(f"test_{k}", v, on_epoch=True, sync_dist=True)

            # Generate videos
            video_cfg = self.eval_config.video
            if video_cfg.enabled:
                seq_solver = self.trainer.datamodule.train_set.load_sequence(
                    start_frame=video_cfg.warmup,
                    len=video_cfg.num_frames,
                )
                cond = seq_solver[: self.ctx_len].to(self.device).unsqueeze(0)
                seq_model = self.generate_sequence(
                    cond=cond,
                    seq_len=video_cfg.num_frames,
                    num_steps=self.num_steps_eval,
                )
                seq_model = seq_model[0].cpu().numpy()
                log_samples_video(
                    data=seq_model,
                    channel_titles=self.eval_config.channel_titles,
                    module=self,
                    fps=video_cfg.fps,
                    normalize=video_cfg.normalize,
                )
                log_comparison_video(
                    gen=seq_model,
                    gt=seq_solver.cpu().numpy(),
                    channel_titles=self.eval_config.channel_titles,
                    module=self,
                    fps=video_cfg.fps,
                )

            # Debug denoising
            denoising_cfg = self.eval_config.debugging.denoising
            if denoising_cfg.enabled:
                cond = self.trainer.datamodule.train_set.load_sequence(
                    start_frame=500,
                    len=self.ctx_len,
                ).to(self.device).unsqueeze(0)

                log_denoising_video(
                    cond=cond,
                    channel_titles=self.eval_config.channel_titles,
                    module=self,
                    fps=denoising_cfg.fps,
                    frames=denoising_cfg.steps,
                    normalize=denoising_cfg.normalize,
                )
        return None

    def on_train_start(self):
        """Log noise schedules to W&B at the start of training."""
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.arange(self.timesteps)

        ax.plot(steps, self.betas.cpu().numpy(), label='betas')
        ax.plot(steps, self.alphas_cumprod.cpu().numpy(), label='alphas_cumprod')
        ax.plot(steps, self.sqrt_alphas_cumprod.cpu().numpy(), label='sqrt_alphas_cumprod')
        ax.plot(steps, self.sqrt_one_minus_alphas_cumprod.cpu().numpy(), label='sqrt_one_minus_alphas_cumprod')
        ax.plot(steps, self.posterior_var.cpu().numpy(), label='posterior_var')

        ax.set_title(f'Noise Schedule: {self.beta_schedule}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        self.logger.experiment.log({"model/noise_schedule": wandb.Image(fig_to_image(fig))})
        plt.close(fig)

    def configure_optimizers(self):
        """Configure optimizers for training."""
        return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
