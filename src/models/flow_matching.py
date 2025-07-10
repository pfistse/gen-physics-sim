import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Optional
import logging
import hydra
from omegaconf import DictConfig

from utils.log import get_logger, log_dict
from utils.wandb import (
    log_samples,
    log_samples_video,
    log_comparison_video,
    log_temporal_deviation,
)
from utils.evaluation import calculate_temporal_deviation

# Create logger for this module
logger = get_logger("models.fm")


class FlowMatchingModel(pl.LightningModule):
    """Flow matching model for physics simulations."""

    def __init__(
        self,
        network: nn.Module,
        dimension: int,
        data_size: List[int],
        sim_fields: List[str],
        sim_params: List[str],
        conditioning_length: int,
        sigma_min: float = 0.001,
        num_steps_eval: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        eval_config: dict = None,
    ):
        """Instantiate the model and store hyperparameters."""
        super(FlowMatchingModel, self).__init__()

        self.save_hyperparameters(ignore=["network"])

        self.unet = network

        self.dimension = dimension
        self.data_size = data_size
        self.sim_fields = sim_fields
        self.sim_params = sim_params
        self.conditioning_length = conditioning_length
        self.target_length = 1
        self.sigma_min = sigma_min
        self.num_steps_eval = num_steps_eval
        self.lr = lr
        self.weight_decay = weight_decay
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

    def compute_vector_field(self, cond: torch.Tensor, data: torch.Tensor):
        """Compute vector fields for training.

        cond: [B, S, C, H, W]
        data: [B, S, C, H, W]
        """
        if self.dimension == 3:
            raise NotImplementedError("3D flow matching not implemented yet")

        device = data.device

        # Always expect 5D tensors [B, S, C, H, W]
        assert (
            len(data.shape) == 5
        ), f"Expected 5D tensor [B, S, C, H, W], got {data.shape}"
        assert (
            len(cond.shape) == 5
        ), f"Expected 5D tensor [B, S, C, H, W], got {cond.shape}"
        assert (
            data.shape[1] == 1
        ), f"Expected single target frame, got {data.shape[1]} frames"

        batch_size = data.shape[0]
        cond_seq_len = cond.shape[1]

        # [B, 1, C, H, W] -> [B, C, H, W]
        x_1 = data.squeeze(1)

        # [B, S_cond, C, H, W] -> [B, S_cond*C, H, W]
        cond = cond.view(
            batch_size, cond_seq_len * cond.shape[2], cond.shape[3], cond.shape[4]
        )

        # Generate noise and time
        x_0 = torch.randn_like(x_1)
        t = torch.rand((batch_size,), device=device)

        # Flow matching computations
        x_t = self.phi_t(x_0, x_1, t)
        v_t = self.u_t(x_0, x_1, t)

        # UNet prediction
        x_t_cond = torch.cat((cond, x_t), dim=1)
        v_t_pred_full = self.unet(x_t_cond, t)

        # Split prediction into conditioning and target parts
        cond_pred = v_t_pred_full[:, : cond.shape[1], :, :]
        v_t_pred = v_t_pred_full[:, cond.shape[1] :, :, :]

        # [B, C, H, W] -> [B, 1, C, H, W]
        v_t_target = v_t.unsqueeze(1)
        v_t_pred = v_t_pred.unsqueeze(1)
        cond_pred = cond_pred.unsqueeze(1)
        cond_target = cond.unsqueeze(1)

        return v_t_target, v_t_pred, cond_target, cond_pred

    def generate_samples(
        self,
        cond: torch.Tensor,
        shape: torch.Size = None,
        num_steps: int = None,
    ):
        """Generate a single frame from ``cond``.

        cond: [B, S, C, H, W]
        return: [B, 1, C, H, W]
        """
        assert (
            len(cond.shape) == 5
        ), f"Expected tensor [B, S, C, H, W], got {cond.shape}"

        device = cond.device
        batch_size = cond.shape[0]
        cond_seq_len = cond.shape[1]

        # [B, S_cond, C, H, W] -> [B, S_cond*C, H, W]
        cond = cond.view(
            batch_size,
            cond_seq_len * cond.shape[2],
            cond.shape[3],
            cond.shape[4],
        )

        # Initialize single frame with noise
        x_0 = torch.randn(
            batch_size,
            self.target_channels,
            cond.shape[3],
            cond.shape[4],
            device=device,
        )
        steps = num_steps or 1
        dt = 1.0 / steps

        with torch.no_grad():

            def wrapper(t, x_t):
                t = torch.tensor([t], dtype=x_t.dtype, device=x_t.device)
                x_t_cond = torch.cat((cond, x_t), dim=1)
                v_t_pred = self.unet(x_t_cond, t)
                return v_t_pred[:, cond.shape[1] :]

            x_1 = self.integrate_euler(f=wrapper, x_0=x_0, t_0=0.0, t_1=1.0, dt=dt)

            # [B, C, H, W] -> [B, 1, C, H, W]
            return x_1.unsqueeze(1)

    def generate_sequence_samples(
        self,
        cond: torch.Tensor,
        num_frames: int,
        num_steps: int = None,
    ):
        """Generate ``num_frames`` frames sequentially.

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
                # Remove oldest frame and add new frame, keeping the window size constant
                current_cond = torch.cat(
                    [current_cond[:, 1:], next_frame], dim=1
                )
            else:
                current_cond = next_frame

        return torch.cat(generated_frames, dim=1)  # [B, num_frames, C, H, W]

    def prediction_step(
        self,
        cond: torch.Tensor,
        shape: torch.Size = None,
        num_steps: int = None,
    ):
        """Prediction step for inference mode."""
        return self.generate_samples(cond, num_steps=num_steps)

    def forward(
        self,
        cond: torch.Tensor,
        data: torch.Tensor = None,
        num_steps: int = None,
    ):
        """Run training or inference depending on ``data``.

        cond: [B, S, C, H, W]
        data: optional [B, 1, C, H, W]
        """
        if data is not None:
            return self.compute_vector_field(cond, data)
        else:
            return self.prediction_step(cond, num_steps=num_steps)

    def training_step(self, batch, batch_idx):
        """Training step for Lightning module."""
        cond = batch[0]
        target = batch[1]

        v_t_target, v_t_pred, cond_target, cond_pred = self.compute_vector_field(
            cond, target
        )
        loss_target = torch.nn.functional.mse_loss(v_t_pred, v_t_target)
        loss_cond = torch.nn.functional.mse_loss(cond_pred, cond_target)
        loss = loss_target  # + loss_cond

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning module."""
        cond = batch[0]
        target = batch[1]

        v_t_target, v_t_pred, cond_target, cond_pred = self.compute_vector_field(
            cond, target
        )
        loss_target = torch.nn.functional.mse_loss(v_t_pred, v_t_target)
        loss_cond = torch.nn.functional.mse_loss(cond_pred, cond_target)
        val_loss = loss_target + loss_cond

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

        v_t_target, v_t_pred, cond_target, cond_pred = self.compute_vector_field(
            cond, target
        )
        loss_target = torch.nn.functional.mse_loss(v_t_pred, v_t_target)
        loss_cond = torch.nn.functional.mse_loss(cond_pred, cond_target)
        test_loss = loss_target + loss_cond

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
                eval_steps = self.num_steps_eval
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
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
