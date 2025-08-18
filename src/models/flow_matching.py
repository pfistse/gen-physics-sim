import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Optional, Dict
from dataclasses import dataclass, field, fields

from utils.log import get_logger
from utils.wandb import (
    log_sample,
    log_samples_video,
    log_comparison_video,
    log_mse,
)
from utils.metrics import calculate_mse

logger = get_logger("models.fm")


@dataclass
class FlowMatchingModelCfg:
    dim: int
    sim_fields: List[str]
    sim_params: List[str]
    ctx_len: int
    net: nn.Module
    sigma_min: float = 0.001
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_steps_eval: int = 1
    eval_config: Dict = field(default_factory=dict)


class FlowMatchingModel(pl.LightningModule):
    """Flow matching model for physics simulations."""

    def __init__(self, cfg: FlowMatchingModelCfg):
        """Instantiate the model with a structured config."""
        super().__init__()

        self.net = cfg.net
        for f in fields(cfg):
            if f.name != "net":
                setattr(self, f.name, getattr(cfg, f.name))

        self.save_hyperparameters(
            {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "net"}
        )

        field_channels = 0
        for field in self.sim_fields:
            if field == "vel":
                field_channels += self.dim
            else:
                field_channels += 1

        self.target_channels = field_channels + len(self.sim_params)
        self.cond_channels = self.ctx_len * self.target_channels

    def phi_t(self, x_0, x_1, t):
        """Interpolate between ``x_0`` and ``x_1`` at time ``t``."""
        t = t.view(-1, 1, 1, 1)
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def u_t(self, x_0, x_1, t):
        """Return the vector field at time ``t``."""
        return x_1 - (1 - self.sigma_min) * x_0

    @staticmethod
    def integrate_euler(f, x_0, t_0, t_1, dt):
        """Simple Euler ODE solver."""
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
    def integrate_rk4(f, x_0, t_0, t_1, dt):
        """Fourth-order Runge-Kutta ODE solver."""
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

    def flow_matching_loss(self, target: torch.Tensor, cond: torch.Tensor, debug: bool = False):
        """Return the training loss for a batch.

        target: [B, 1, C, H, W]
        cond: [B, S, C, H, W]
        """
        assert len(target.shape) == 5, f"Expected [B, 1, C, H, W], got {target.shape}"
        assert len(cond.shape) == 5, f"Expected [B, S, C, H, W], got {cond.shape}"
        assert target.size(1) == 1, "Second dimension of `target` must be 1"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)
        x_1 = target.squeeze(1)

        x_0 = torch.randn_like(x_1)
        t = torch.rand((B,), device=target.device)

        x_t = self.phi_t(x_0, x_1, t)
        v_t = self.u_t(x_0, x_1, t)

        x_in = torch.cat([cond_flat, x_t], dim=1)
        v_pred_full = self.net(x_in, t)
        cond_pred = v_pred_full[:, : cond_flat.shape[1], :, :]
        v_pred = v_pred_full[:, cond_flat.shape[1]:, :, :]

        loss_target = F.mse_loss(v_pred, v_t)
        loss_cond = F.mse_loss(cond_pred, cond_flat)
        loss = loss_target + loss_cond

        if debug:
            return loss, v_pred.unsqueeze(1), v_t.unsqueeze(1), cond_pred.unsqueeze(1), cond_flat.unsqueeze(1)
        return loss

    def generate_samples(self, cond: torch.Tensor, num_steps: int = None):
        """Sample the next frame from ``cond``.

        cond: [B, S, C, H, W]
        return: [B, 1, C, H, W]
        """
        assert len(cond.shape) == 5, f"Expected [B, S, C, H, W], got {cond.shape}"

        B, S, C, H, W = cond.shape
        cond_flat = cond.view(B, S * C, H, W)

        x = torch.randn(B, self.target_channels, H, W, device=cond.device)
        steps = num_steps if num_steps is not None else 1
        dt = 1.0 / steps

        with torch.no_grad():

            def wrapper(t, x_t):
                t = torch.tensor([t], dtype=x_t.dtype, device=x_t.device)
                x_in = torch.cat([cond_flat, x_t], dim=1)
                v_pred = self.net(x_in, t)
                return v_pred[:, cond_flat.shape[1]:]

            x_1 = self.integrate_euler(wrapper, x, 0.0, 1.0, dt)
            return x_1.unsqueeze(1)

    def generate_sequence(self, cond: torch.Tensor, seq_len: int, num_steps: int = None):
        """Generate ``seq_len`` sequential samples."""

        B, S, C, H, W = cond.shape
        field_channels = self.target_channels - len(self.sim_params)
        const_params = None
        if len(self.sim_params) > 0:
            const_params = cond[:, :1, field_channels:, :, :]

        gen_seq = []
        cond_window = cond.clone()
        for _ in range(seq_len):
            next_frame = self.generate_samples(cond_window, num_steps=num_steps)
            if const_params is not None:
                next_frame[:, :, field_channels:, :, :] = const_params
            gen_seq.append(next_frame)
            if S > 1:
                cond_window = torch.cat([cond_window[:, 1:], next_frame], dim=1)
            else:
                cond_window = next_frame
        return torch.cat(gen_seq, dim=1)

    def prediction_step(
        self,
        cond: torch.Tensor,
        num_steps: int = None,
    ):
        """Prediction step for inference mode."""
        return self.generate_samples(cond, num_steps=num_steps)

    def forward(
        self,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: int = None,
    ):
        """Run training or inference depending on ``target``."""
        if target is not None:
            return self.flow_matching_loss(target, cond)
        return self.prediction_step(cond, num_steps=num_steps)

    def training_step(self, batch, batch_idx):
        """Training step for Lightning module."""
        cond, target = batch
        loss = self.flow_matching_loss(target, cond)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning module."""
        cond, target = batch
        loss = self.flow_matching_loss(target, cond)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.logger is not None:
            with torch.no_grad():
                eval_steps = self.num_steps_eval
                samples = self.generate_samples(cond, num_steps=eval_steps)

                log_sample(
                    samples=samples[0, 0].cpu().numpy(),
                    channel_titles=self.eval_config.channel_titles,
                    module=self,
                )

                mse_cfg = self.eval_config.metrics.mse
                if mse_cfg.enabled:
                    metrics = calculate_mse(
                        model=self,
                        num_simulations=mse_cfg.num_sims,
                        num_time_steps=mse_cfg.num_time_steps,
                        start_frame=mse_cfg.start_frame,
                        channel_titles=self.eval_config.channel_titles,
                        num_steps=eval_steps,
                    )
                    log_mse(metrics, self, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for Lightning module."""
        cond, target = batch
        loss = self.flow_matching_loss(target, cond)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            eval_steps = self.num_steps_eval
            mse_cfg = self.eval_config.metrics.mse
            if mse_cfg.enabled:
                metrics = calculate_mse(
                    model=self,
                    num_simulations=mse_cfg.num_sims,
                    num_time_steps=mse_cfg.num_time_steps,
                    start_frame=mse_cfg.start_frame,
                    channel_titles=self.eval_config.channel_titles,
                    num_steps=eval_steps,
                )
                log_mse(metrics, self, prefix="test")

            video_cfg = self.eval_config.video
            if video_cfg.enabled:
                seq = self.generate_sequence(
                    cond=cond[:1],
                    seq_len=video_cfg.num_frames,
                    num_steps=eval_steps,
                )
                video_array = seq[0].cpu().numpy()
                log_samples_video(
                    data=video_array,
                    channel_titles=self.eval_config.channel_titles,
                    module=self,
                    fps=video_cfg.fps,
                    normalize=video_cfg.normalize,
                )

                datamodule = self.trainer.datamodule
                ds = datamodule.test_dataset
                first_idx = batch_idx * datamodule.batch_size
                _, start_frame, folder = ds.sequence_paths[first_idx]
                sim_num = int(folder.split('_')[1])
                gt_seq = ds.load_sequence(
                    sim=sim_num,
                    start_frame=start_frame,
                    len=video_cfg.num_frames,
                )
                gt_array = gt_seq[ds.ctx_len:].cpu().numpy()
                log_comparison_video(
                    gen=video_array[: len(gt_array)],
                    gt=gt_array,
                    channel_titles=self.eval_config.channel_titles,
                    module=self,
                    fps=video_cfg.fps,
                )

        return loss

    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
