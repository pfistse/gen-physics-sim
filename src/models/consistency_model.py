import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple, List, Union, Optional
import logging
import numpy as np
import copy
import wandb

from utils.log import get_logger, log_dict
from utils.wandb import (
    log_samples,
    log_samples_video,
    log_comparison_video,
    log_temporal_deviation,
)
from utils.evaluation import calculate_temporal_deviation

logger = get_logger("models.cm")


class ConsistencyModel(pl.LightningModule):
    """Consistency model for physics simulations."""

    def __init__(
        self,
        dimension: int,
        data_size: List[int],
        sim_fields: List[str],
        sim_params: List[str],
        conditioning_length: int,
        network: nn.Module,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        ema_rate: float = 0.9999,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        num_steps: int = 40,
        num_steps_eval: int = 1,
        consistency_training: str = "ct",
        teacher_model_path: Optional[str] = None,
        eval_config: Dict = None,
        unroll_steps: int = 1,
        target_loss_weight: float = 0.1,
    ):
        """Instantiate the model and store hyperparameters."""
        super(ConsistencyModel, self).__init__()

        self.save_hyperparameters(ignore=["network"])

        self.dimension = dimension
        self.data_size = data_size
        self.sim_fields = sim_fields
        self.sim_params = sim_params
        self.conditioning_length = conditioning_length
        self.target_length = 1
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.ema_rate = ema_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.num_steps_eval = num_steps_eval
        self.consistency_training = consistency_training
        self.teacher_model_path = teacher_model_path
        self.unroll_steps = unroll_steps
        self.target_loss_weight = target_loss_weight

        self.eval_config = eval_config or {}

        field_channels = 0
        for field in sim_fields:
            if field == "vel":
                field_channels += dimension
            elif field == "pres":
                field_channels += 1
            else:
                field_channels += 1

        self.target_channels = field_channels + len(sim_params)

        cond_frames = conditioning_length
        channels_per_frame = field_channels + len(sim_params)
        self.cond_channels = cond_frames * channels_per_frame
        self.total_channels = self.cond_channels + self.target_channels
        self.model = network

        # Initialize target network (EMA of online network)
        self.target_model = self._create_target_model()
        self._initialize_target_model()

        # Load teacher model if doing consistency distillation
        self.teacher_model = None
        if consistency_training == "cd" and teacher_model_path:
            self.teacher_model = self._load_teacher_model(teacher_model_path)

    def _create_target_model(self):
        """Create target network with same architecture as online network."""
        target_model = copy.deepcopy(self.model)

        # Disable gradients
        for param in target_model.parameters():
            param.requires_grad = False

        return target_model

    def _initialize_target_model(self):
        """Initialize target model with online model parameters."""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.model.parameters()
            ):
                target_param.data.copy_(online_param.data)

    def _update_target_model(self):
        """Update target model using exponential moving average."""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_model.parameters(), self.model.parameters()
            ):
                target_param.data.mul_(self.ema_rate).add_(
                    online_param.data, alpha=1 - self.ema_rate
                )

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

        t = (
            sigma_max_rho
            + step_indices / (num_steps - 1) * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho
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
        cond: [B, S_cond*C, H, W]
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

        # Only process samples above sigma_min
        if process_mask.sum() > 0:
            c_skip_val = self.c_skip(sigma_reshaped)
            c_out_val = self.c_out(sigma_reshaped)
            c_in_val = self.c_in(sigma_reshaped)
            c_noise_val = self.c_noise(sigma).view(-1)

            x_in = torch.cat([cond, c_in_val * x], dim=1)

            model = self.target_model if use_target else self.model

            f_theta_full = model(x_in, c_noise_val)

            f_theta_cond = f_theta_full[:, : cond.shape[1], :, :]
            f_theta = f_theta_full[:, cond.shape[1] :, :, :]

            consistency_output = c_skip_val * x + c_out_val * f_theta

            # Apply the process mask to only update non-boundary samples
            output = boundary_mask * x + process_mask * consistency_output

        if return_cond:
            return output, f_theta_cond
        else:
            return output

    def consistency_loss(self, target_seq: torch.Tensor, cond_seq: torch.Tensor):
        """Return the training loss for a batch.

        target_seq: [B, S, C, H, W]
        cond_seq: [B, S, C, H, W]
        """

        assert (
            self.unroll_steps == target_seq.shape[1]
        ), "unroll_steps must match target sequence length"

        device = target_seq.device
        s_batch = target_seq.shape[0]
        s_channels = cond_seq.shape[2]
        num_steps = self.num_steps
        num_unroll_steps = self.unroll_steps

        t_schedule = self.get_noise_schedule(num_steps).to(device)

        total_loss = 0.0

        cond = cond_seq.view(
            s_batch,
            cond_seq.size(1) * cond_seq.size(2),
            cond_seq.size(3),
            cond_seq.size(4),
        )  # [B, S_cond*C, H, W]

        for i in range(num_unroll_steps):
            target = target_seq[:, i]

            indices = torch.randint(0, num_steps - 1, (s_batch,), device=device)
            sigma_t = t_schedule[indices]  # Current timestep (higher noise)
            sigma_s = t_schedule[indices + 1]  # Next timestep (lower noise)

            noise = torch.randn_like(target)
            x_t = target + sigma_t.view(-1, 1, 1, 1) * noise
            x_s = target + sigma_s.view(-1, 1, 1, 1) * noise

            with torch.no_grad():
                f_s, _ = self.consistency_function(
                    x_s, sigma_s, cond, use_target=True, return_cond=True
                )

            f_t, cond_t = self.consistency_function(
                x_t, sigma_t, cond, use_target=False, return_cond=True
            )

            loss_consistency = torch.nn.functional.mse_loss(f_t, f_s)
            loss_cond = torch.nn.functional.mse_loss(cond_t, cond)

            # Improving stability
            target_loss = (
                torch.nn.functional.mse_loss(f_t, target) * self.target_loss_weight
            )

            total_loss += loss_consistency + loss_cond + target_loss

            if cond_seq.shape[1] > 1:
                cond = torch.cat([cond[:, s_channels:], f_t], dim=1)
            else:
                cond = f_t

        return total_loss / num_unroll_steps

    def generate_samples(
        self,
        cond_seq: torch.Tensor,
        num_steps: int = 1,
        use_ema: bool = True,
        requires_grad: bool = False,
    ):
        """Sample the next frame from ``cond_seq``.

        cond_seq: [B, S, C, H, W]
        return: [B, 1, C, H, W]
        """
        if self.dimension == 3:
            raise NotImplementedError("3D consistency model not implemented yet")

        device = cond_seq.device

        assert (
            len(cond_seq.shape) == 5
        ), f"Expected tensor [B, S, C, H, W], got {cond_seq.shape}"

        batch_size = cond_seq.shape[0]
        cond_seq_len = cond_seq.shape[1]

        # [B, S_cond, C, H, W] -> [B, S_cond*C, H, W]
        cond = cond_seq.view(
            batch_size,
            cond_seq_len * cond_seq.shape[2],
            cond_seq.shape[3],
            cond_seq.shape[4],
        )

        model_use_target = use_ema
        context = torch.enable_grad if requires_grad else torch.no_grad
        with context():
            if num_steps == 1:
                # Single-step generation
                x = torch.randn(
                    batch_size,
                    self.target_channels,
                    cond_seq.shape[3],
                    cond_seq.shape[4],
                    device=device,
                )
                sigma = torch.full((batch_size,), self.sigma_max, device=device)

                generated = self.consistency_function(
                    x, sigma, cond, use_target=model_use_target
                )
            else:
                # Multi-step generation
                t_schedule = self.get_noise_schedule(num_steps).to(device)

                sigma = t_schedule[0]
                x = (
                    torch.randn(
                        batch_size,
                        self.target_channels,
                        cond_seq.shape[3],
                        cond_seq.shape[4],
                        device=device,
                    )
                    * sigma
                )

                for i in range(num_steps - 1):
                    sigma = t_schedule[i]
                    x0 = self.consistency_function(
                        x, sigma, cond, use_target=model_use_target
                    )

                    next_sigma = t_schedule[i + 1]
                    next_sigma = torch.clamp(
                        next_sigma, min=self.sigma_min, max=self.sigma_max
                    )
                    noise_scale = torch.sqrt(
                        torch.clamp(next_sigma**2 - self.sigma_min**2, min=0.0)
                    )
                    noise = torch.randn_like(x)
                    x = x0 + noise_scale * noise

                generated = x

            # [B, C, H, W] -> [B, 1, C, H, W]
            return generated.unsqueeze(1)

    def generate_sequence_samples(
        self,
        cond: torch.Tensor,
        num_frames: int,
        num_steps: int = 1,
    ):
        """Generate ``num_frames`` sequential samples.

        cond: [B, S_cond, C, H, W]
        return: [B, num_frames, C, H, W]
        """

        num_cond_frames = cond.shape[1]

        generated_frames = []
        current_cond = cond.clone()
        field_channels = self.target_channels - len(self.sim_params)
        const_params = None
        if len(self.sim_params) > 0:
            const_params = cond[:, 0, field_channels:, :, :].unsqueeze(1)

        for _ in range(num_frames):
            next_frame = self.generate_samples(
                current_cond, num_steps=num_steps
            )  # [B, 1, C, H, W]
            if const_params is not None:
                next_frame_fields = next_frame[:, :, :field_channels, :, :]
                repeated_params = const_params.expand(
                    next_frame.size(0), 1, -1, next_frame.size(-2), next_frame.size(-1)
                )
                next_frame = torch.cat([next_frame_fields, repeated_params], dim=2)
            generated_frames.append(next_frame)

            if num_cond_frames > 1:
                # Remove oldest frame and add new frame
                current_cond = torch.cat(
                    [current_cond[:, 1:], next_frame], dim=1
                )
            else:
                current_cond = next_frame

        return torch.cat(generated_frames, dim=1)  # [B, num_frames, C, H, W]

    def prediction_step(self, cond: torch.Tensor, num_steps: int = 1):
        """Prediction step for inference mode."""
        return self.generate_samples(cond, num_steps=num_steps)

    def forward(
        self,
        cond: torch.Tensor,
        target: torch.Tensor = None,
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
            return self.prediction_step(cond, num_steps=num_steps)

    def training_step(self, batch, batch_idx):
        """Training step for Lightning module."""
        cond = batch[0]
        target = batch[1]

        loss = self.forward(cond, target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA target network after optimizer step."""
        self._update_target_model()

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning module."""
        cond = batch[0]
        target = batch[1]

        val_loss = self.forward(cond, target)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        if batch_idx == 0 and self.logger is not None:
            with torch.no_grad():
                eval_steps = self.num_steps_eval
                generated_samples = self.generate_samples(
                    cond, num_steps=eval_steps
                )

                log_samples(
                    samples=generated_samples[0, 0].cpu().numpy(),
                    channel_titles=self.eval_config["channel_titles"],
                    lightning_module=self,
                )

                # Calulate metrics
                temp_deviation_config = self.eval_config["metrics"]["temp_deviation"]
                if temp_deviation_config["enabled"]:
                    deviation_metrics = calculate_temporal_deviation(
                        model=self,
                        num_simulations=temp_deviation_config["num_simulations"],
                        num_time_steps=temp_deviation_config["num_time_steps"],
                        start_frame=temp_deviation_config.get("start_frame"),
                        channel_titles=self.eval_config["channel_titles"],
                        num_steps=eval_steps,
                    )

                    log_temporal_deviation(deviation_metrics, self, prefix="val")

        return val_loss

    def test_step(self, batch, batch_idx):
        """Test step for Lightning module."""
        cond = batch[0]
        target = batch[1]

        test_loss = self.forward(cond, target)
        self.log("test_loss", test_loss)

        if batch_idx == 0:
            # Calulate metrics
            temp_deviation_config = self.eval_config["metrics"]["temp_deviation"]
            if temp_deviation_config["enabled"]:
                eval_steps = self.num_steps_eval
                deviation_metrics = calculate_temporal_deviation(
                    model=self,
                    num_simulations=temp_deviation_config["num_simulations"],
                    num_time_steps=temp_deviation_config["num_time_steps"],
                    start_frame=temp_deviation_config.get("start_frame"),
                    channel_titles=self.eval_config["channel_titles"],
                    num_steps=eval_steps,
                )

                log_temporal_deviation(deviation_metrics, self, prefix="test")

            # Generate videos
            video_config = self.eval_config["video"]
            if video_config["enabled"]:
                video_samples = self.generate_sequence_samples(
                    cond=cond[:1],
                    num_frames=video_config["num_frames"],
                    num_steps=eval_steps,
                )

                video_array = video_samples[0].cpu().numpy()

                log_samples_video(
                    samples=video_array,
                    channel_titles=self.eval_config["channel_titles"],
                    lightning_module=self,
                    fps=video_config["fps"],
                )

                datamodule = self.trainer.datamodule
                ds = datamodule.test_dataset
                first_idx = batch_idx * datamodule.batch_size
                _, start_frame, folder = ds.sequence_paths[first_idx]
                seq = ds.load_sequence(
                    sim_folder=folder,
                    start_frame=start_frame,
                    target_length=video_config["num_frames"],
                )
                if seq is not None:
                    gt_array = seq[1].cpu().numpy()
                    log_comparison_video(
                        generated=video_array[: len(gt_array)],
                        target=gt_array,
                        channel_titles=self.eval_config["channel_titles"],
                        lightning_module=self,
                        fps=video_config["fps"],
                    )

        return test_loss

    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
