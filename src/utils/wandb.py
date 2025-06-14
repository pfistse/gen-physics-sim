import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
import pytorch_lightning as pl
from io import BytesIO
from typing import Dict, List
from utils.log import get_logger


def _compute_vorticity(vy: np.ndarray, vx: np.ndarray) -> np.ndarray:
    """Compute 2D vorticity from velocity components."""
    vy_dy, vy_dx = np.gradient(vy)
    vx_dy, vx_dx = np.gradient(vx)
    return vx_dy - vy_dx

ICEFIRE_CMAP = sns.color_palette("icefire", as_cmap=True)
logger = get_logger("utils.wandb")


def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        buffer.seek(0)
        return PIL.Image.open(buffer)
    except Exception:
        fig.canvas.draw()
        return PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.renderer.buffer_rgba()
        )


def log_samples(
    samples: np.ndarray,
    channel_titles: List[str],
    lightning_module: pl.LightningModule,
):
    """Log physics samples to wandb."""
    try:
        assert (
            len(samples.shape) == 3
        ), f"Expected 3D array (channels, height, width), got {samples.shape}"

        images_to_log = {}
        num_channels = samples.shape[0]

        for channel_idx in range(min(num_channels, len(channel_titles))):
            title = channel_titles[channel_idx]

            if title is None:
                continue

            field_data = np.nan_to_num(samples[channel_idx])

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(field_data, cmap=ICEFIRE_CMAP, aspect="equal")
            plt.colorbar(im, ax=ax)

            images_to_log[f"generated/frame_{title}"] = wandb.Image(fig_to_image(fig))
            plt.close(fig)
            
        vort = _compute_vorticity(samples[0], samples[1])
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(vort, cmap=ICEFIRE_CMAP, aspect="equal")
        plt.colorbar(im, ax=ax)
        images_to_log["generated/frame_vorticity"] = wandb.Image(fig_to_image(fig))
        plt.close(fig)

        if images_to_log:
            for key, wandb_image in images_to_log.items():
                lightning_module.logger.log_image(key=key, images=[wandb_image.image])

    except Exception as e:
        logger.error(f"Error logging samples: {e}")


def log_samples_video(
    samples: np.ndarray,
    channel_titles: List[str],
    lightning_module: pl.LightningModule,
    fps: int = 10,
):
    """Log physics samples as video to wandb."""
    try:
        assert (
            len(samples.shape) == 4
        ), f"Expected 4D array (num_frames, channels, height, width), got {samples.shape}"

        num_frames, num_channels, height, width = samples.shape
        videos_to_log = {}

        def create_video(field_data):
            field_data = np.nan_to_num(field_data)
            field_min, field_max = field_data.min(), field_data.max()
            if field_max > field_min:
                field_data = (field_data - field_min) / (field_max - field_min)

            field_rgb = np.zeros((num_frames, 3, height, width), dtype=np.uint8)
            for frame_idx in range(num_frames):
                colored = ICEFIRE_CMAP(field_data[frame_idx])
                field_rgb[frame_idx] = (
                    colored[:, :, :3].transpose(2, 0, 1) * 255
                ).astype(np.uint8)

            return wandb.Video(field_rgb, fps=fps, format="mp4")

        for channel_idx in range(min(num_channels, len(channel_titles))):
            title = channel_titles[channel_idx]

            if title is None:
                continue

            videos_to_log[f"generated/video_{title}"] = create_video(
                samples[:, channel_idx]
            )

        vort_frames = []
        for f in range(num_frames):
            vort = _compute_vorticity(samples[f, 0], samples[f, 1])
            vort_frames.append(vort)
        vort_array = np.stack(vort_frames, axis=0)
        videos_to_log["generated/video_vorticity"] = create_video(vort_array)

        if videos_to_log:
            wandb.log(videos_to_log)

    except Exception as e:
        logger.error(f"Error logging video samples: {e}")


def log_temporal_deviation(
    deviation_metrics: Dict[str, any],
    lightning_module: pl.LightningModule,
    prefix: str = "test",
):
    """Log deviation metrics and plots to wandb."""
    try:
        scalar_metrics = {}

        if "mse_mean" in deviation_metrics:
            scalar_metrics[f"{prefix}/mse_mean"] = deviation_metrics["mse_mean"]

        if "rmse_mean" in deviation_metrics:
            scalar_metrics[f"{prefix}/rmse_mean"] = deviation_metrics["rmse_mean"]

        mse_per_timestep = deviation_metrics.get("mse_per_timestep", [])
        rmse_per_timestep = deviation_metrics.get("rmse_per_timestep", [])
        simulations = deviation_metrics.get("simulations", [])

        def _log_time_plot(values_list, ylabel, key):
            if not values_list:
                return
            arr = np.array(values_list)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            timesteps = np.arange(len(mean))

            ax.fill_between(
                timesteps,
                mean - std,
                mean + std,
                alpha=0.2,
                color="steelblue",
                zorder=1,
                label="±1 Std. Dev.",
            )
            ax.plot(
                timesteps,
                mean,
                color="steelblue",
                linewidth=2.5,
                zorder=2,
                label="Mean",
            )

            ax.set_xlabel("Time Step", fontsize=12, fontweight="normal")
            ax.set_ylabel(ylabel, fontsize=12, fontweight="normal")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_facecolor("white")
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.margins(x=0.02, y=0.05)
            ax.legend(frameon=False, fontsize=10)
            plt.tight_layout()

            scalar_metrics[key] = wandb.Image(fig_to_image(fig))
            plt.close(fig)

        if mse_per_timestep and simulations:
            all_mse_values = [
                sim["mse_per_step"] for sim in simulations if "mse_per_step" in sim
            ]
            _log_time_plot(
                all_mse_values, "Mean Squared Error", f"{prefix}/mse_over_time"
            )

        if rmse_per_timestep and simulations:
            all_rmse_values = [
                sim["rmse_per_step"] for sim in simulations if "rmse_per_step" in sim
            ]
            _log_time_plot(
                all_rmse_values, "Root Mean Squared Error", f"{prefix}/rmse_over_time"
            )

        if scalar_metrics:
            wandb.log(scalar_metrics)

    except Exception as e:
        logger.error(f"Failed to log temporal deviation metrics: {e}")
