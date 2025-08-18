from utils.metrics import calculate_mse, calculate_vrs
from utils.wandb import (
    log_sample,
    log_samples_video,
    log_comparison_video,
    log_mse,
)
from utils.log import get_logger, log_dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple, List, Union, Optional
from dataclasses import dataclass, field, fields
import logging
import numpy as np
import copy
import wandb
import rootutils

# Setup root directory for imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


logger = get_logger("models.cm")


@dataclass
class ConsistencyModelCfg:
    dim: int
    sim_fields: List[str]
    sim_params: List[str]
    ctx_len: int
    net: nn.Module
    sigma_min: float = 0.01
    sigma_max: float = 20.0
    sigma_data: float = 1.0
    rho: float = 7.0
    ema_rate: float = 0.95
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_steps: int = 8
    num_steps_eval: int = 1
    rand_cons_step: bool = True
    consistency_training: str = "ct"
    teacher_model_path: Optional[str] = None
    eval_config: Dict = field(default_factory=dict)
    unroll_steps: int = 1
    target_loss_weight: float = 0.1


class ConsistencyModel(pl.LightningModule):
    """Consistency model for physics simulations."""

    def __init__(self, cfg: ConsistencyModelCfg):
        """Instantiate the model with a structured config."""
        super(ConsistencyModel, self).__init__()

        self.net = cfg.net

        for f in fields(cfg):
            if f.name != 'net':
                setattr(self, f.name, getattr(cfg, f.name))

        # Initialize target network (EMA of online network)
        self.target_model = self._create_target_model()
        self._initialize_target_model()

        # Load teacher model if doing consistency distillation
        self.teacher_model = None
        if self.consistency_training == "cd" and self.teacher_model_path:
            self.teacher_model = self._load_teacher_model(self.teacher_model_path)

        cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "net"}
        self.save_hyperparameters(cfg_dict)

    def _create_target_model(self):
        """Create target network with same architecture as online network."""
        target_model = copy.deepcopy(self.net)

        # Disable gradients
        for param in target_model.parameters():
            param.requires_grad = False

        return target_model

    def _initialize_target_model(self):
        """Initialize target model with online model parameters."""
        with torch.no_grad():
            for target_param, online_param in zip(self.target_model.parameters(), self.net.parameters()):
                target_param.data.copy_(online_param.data)

    def _update_target_model(self):
        """Update target model using exponential moving average."""
        with torch.no_grad():
            for target_param, online_param in zip(self.target_model.parameters(), self.net.parameters()):
                target_param.data.mul_(self.ema_rate).add_(online_param.data, alpha=1 - self.ema_rate)

    def _load_teacher_model(self, teacher_path: str):
        """Load teacher model for consistency distillation."""
        logger.info(f"Loading teacher model from {teacher_path}")
        # Placeholder until teacher model loading is implemented
        return None

    def get_noise_schedule(self, num_steps: int):
        """Generate noise schedule for consistency training following the paper."""
        step_indices = torch.arange(num_steps, dtype=torch.float32)

        # Paper's schedule: sigma(i) = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)

        t = (sigma_max_rho + step_indices / (num_steps - 1) * (sigma_min_rho - sigma_max_rho)) ** self.rho
        t = torch.clamp(t, min=self.sigma_min, max=self.sigma_max)

        return t

    def c_skip(self, sigma):
        """Skip connection scaling function."""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        """Output scaling function."""
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma):
        """Input scaling function."""
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        """Noise conditioning scaling function."""
        sigma_clamped = torch.clamp(sigma, min=self.sigma_min)
        return 0.25 * torch.log(sigma_clamped)

    def consistency_function(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        use_target: bool = False,
        return_cond: bool = False,
    ):
        """Denoise ``x`` at noise ``sigma`` with ``cond``.

        x: [B, C, H, W]
        sigma: [B] or scalar
        cond: [B, S*C, H, W]
        return: [B, C, H, W]
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(x.shape[0])
        elif sigma.dim() == 1 and sigma.shape[0] != x.shape[0]:
            sigma = sigma.expand(x.shape[0])

        sigma_reshaped = sigma.view(-1, 1, 1, 1)

        # Boundary condition: when sigma <= sigma_min, return x (no denoising)
        boundary_mask = (sigma <= self.sigma_min).float().view(-1, 1, 1, 1)

        # Only process samples that are above the boundary
        process_mask = (sigma > self.sigma_min).float().view(-1, 1, 1, 1)

        # Initialize output with input (identity for boundary condition)
        output = x.clone()

        # if process_mask.sum() > 0: TODO
        c_skip_val = self.c_skip(sigma_reshaped)
        c_out_val = self.c_out(sigma_reshaped)
        c_in_val = self.c_in(sigma_reshaped)
        c_noise_val = self.c_noise(sigma).view(-1)

        x_in = torch.cat([cond, c_in_val * x], dim=1)

        model = self.target_model if use_target else self.net

        f_theta_full = model(x_in, c_noise_val)

        f_theta_cond = f_theta_full[:, : cond.shape[1], :, :]
        f_theta = f_theta_full[:, cond.shape[1]:, :, :]

        consistency_output = c_skip_val * x + c_out_val * f_theta

        # Apply the process mask to only update non-boundary samples
        output = boundary_mask * x + process_mask * consistency_output

        if return_cond:
            return output, f_theta_cond
        else:
            return output

    def consistency_loss(self, target: torch.Tensor, cond: torch.Tensor):
        """Return the training loss for a batch.

        target: [B, U, C, H, W]
        cond: [B, S, C, H, W]
        return: scalar loss tensor
        """

        assert len(target.shape) == 5, f"Expected target tensor [B, U, C, H, W], got {target.shape}"
        assert len(cond.shape) == 5, f"Expected cond tensor [B, S, C, H, W], got {cond.shape}"
        assert target.size(1) == self.unroll_steps, "Second dimension of `target` must match `unroll_steps`"

        device = target.device
        B, S, C, H, W = cond.shape
        U = target.size(1)
        T = self.num_steps

        # Noise level increases with step index
        t_schedule = self.get_noise_schedule(T).to(device)

        total_loss = 0.0

        cond_flat = cond.view(B, S * C, H, W)

        for i in range(U):
            target_frame = target[:, i]

            if self.rand_cons_step:
                # Choose random consistency steps
                t_idx = torch.randint(0, T - 1, (B,), device=device)
                range_len = (T - 1) - t_idx
                u = torch.rand(B, device=device)
                offset = (u * range_len.float()).floor().long()
                s_idx = t_idx + 1 + offset
            else:
                # Choose adjacent consistency steps
                t_idx = torch.randint(0, T - 1, (B,), device=device)
                s_idx = t_idx + 1

            sigma_t = t_schedule[t_idx]  # noise level at t
            sigma_s = t_schedule[s_idx]  # noise level at s (next step)

            noise = torch.randn_like(target_frame)
            x_t = target_frame + sigma_t.view(-1, 1, 1, 1) * noise
            x_s = target_frame + sigma_s.view(-1, 1, 1, 1) * noise

            with torch.no_grad():
                f_s, _ = self.consistency_function(x_s, sigma_s, cond_flat, use_target=True, return_cond=True)

            f_t, cond_t = self.consistency_function(x_t, sigma_t, cond_flat, use_target=False, return_cond=True)

            loss_consistency = torch.nn.functional.mse_loss(f_t, f_s)
            loss_cond = torch.nn.functional.mse_loss(cond_t, cond_flat)

            # Improves stability
            target_loss = torch.nn.functional.mse_loss(f_t, target_frame) * self.target_loss_weight

            total_loss += loss_consistency + loss_cond + target_loss

            # Slide the conditioning window forward
            if cond.shape[1] > 1:
                cond_flat = torch.cat([cond_flat[:, C:], f_t], dim=1)
            else:
                cond_flat = f_t

        return total_loss / U

    def generate_samples(
        self,
        cond: torch.Tensor,
        num_steps: int = 1,
        use_ema: bool = True,
        requires_grad: bool = False,
    ):
        """Sample the next frame from ``cond``.

        cond: [B, S, C, H, W]
        return: [B, 1, C, H, W]
        """
        if self.dim == 3:
            raise NotImplementedError("3D consistency model not implemented yet")

        assert len(cond.shape) == 5, f"Expected cond tensor [B, S, C, H, W], got {cond.shape}"

        device = cond.device
        B, S, C, H, W = cond.shape

        # [B, S, C, H, W] -> [B, S * C, H, W]
        cond_flat = cond.view(B, S * C, H, W)

        context = torch.enable_grad if requires_grad else torch.no_grad
        with context():
            if num_steps == 1:
                # Single-step generation
                x = torch.randn(B, C, H, W, device=device)
                sigma = torch.full((B,), self.sigma_max, device=device)

                x0 = self.consistency_function(x, sigma, cond_flat, use_target=use_ema)
            else:
                # Multi-step generation
                t_schedule = self.get_noise_schedule(num_steps).to(device)

                sigma = t_schedule[0]
                x = torch.randn(B, C, H, W, device=device) * sigma

                for i in range(num_steps - 1):
                    sigma = t_schedule[i]
                    x0 = self.consistency_function(x, sigma, cond_flat, use_target=use_ema)

                    next_sigma = t_schedule[i + 1]
                    next_sigma = torch.clamp(next_sigma, min=self.sigma_min, max=self.sigma_max)
                    noise_scale = torch.sqrt(torch.clamp(sigma**2 - next_sigma**2, min=1e-12))  # TODO check
                    noise = torch.randn_like(x)
                    x = x0 + noise_scale * noise

                x0 = self.consistency_function(x, t_schedule[-1], cond_flat, use_target=use_ema)  # TODO check

            # [B, C, H, W] -> [B, 1, C, H, W]
            return x0.unsqueeze(1)

    def generate_sequence(
        self,
        cond: torch.Tensor,
        seq_len: int,
        num_steps: int = 1,
    ):
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
            next_frame = self.generate_samples(cond_window, num_steps=num_steps)  # [B, 1, C, H, W]

            # Restore parameters
            if const_params is not None:
                next_frame[:, :, -P:, :, :] = const_params

            gen_seq.append(next_frame)

            if cond_len > 1:
                # Remove oldest frame and add new frame
                cond_window = torch.cat([cond_window[:, 1:], next_frame], dim=1)
            else:
                cond_window = next_frame

        return torch.cat(gen_seq, dim=1)  # [B, seq_len, C, H, W]

    def prediction_step(self, cond: torch.Tensor, num_steps: int):
        """Prediction step for inference mode."""
        return self.generate_samples(cond, num_steps=num_steps)

    def forward(
        self,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ):
        """Training and inference entry point.

        cond: [B, S, C, H, W]
        target: optional [B, 1, C, H, W]
        """
        if target is not None:
            # Training mode
            return self.consistency_loss(target, cond)
        else:
            # Inference mode
            return self.prediction_step(cond, num_steps)

    def training_step(self, batch, batch_idx):
        """Training step for Lightning module."""
        cond, target = batch

        loss = self.forward(cond, target, num_steps=1)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA target network after optimizer step."""
        self._update_target_model()

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning module."""

        self.test_step(batch, batch_idx)

        # NOTE validation loss does not matter
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
                    **mse_cfg
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

        return None

    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
