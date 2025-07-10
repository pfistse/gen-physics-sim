import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
import pytorch_lightning as pl
import itertools
import tempfile
import imageio
from io import BytesIO
from typing import Any, Dict, List
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

        # Compute and log vorticity if velocity channels are available
        if num_channels >= 2:
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

        # Add vorticity video if velocity channels are available
        if num_channels >= 2:
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


def log_comparison_video(
    generated: np.ndarray,
    target: np.ndarray,
    channel_titles: List[str],
    lightning_module: pl.LightningModule,
    fps: int = 10,
):
    """Log side-by-side comparison videos of generated and ground truth."""
    try:
        assert generated.shape == target.shape
        assert len(generated.shape) == 4

        num_frames, num_channels, height, width = generated.shape
        videos = {}

        def _norm(data):
            data = np.nan_to_num(data)
            vmin, vmax = data.min(), data.max()
            if vmax > vmin:
                data = (data - vmin) / (vmax - vmin)
            return data

        def _make_video(gen, gt):
            diff = gen - gt
            gen = _norm(gen)
            gt = _norm(gt)
            diff = _norm(diff)

            frames = np.zeros((num_frames, 3, height, width * 3), dtype=np.uint8)
            for i in range(num_frames):
                g = (ICEFIRE_CMAP(gen[i])[..., :3].transpose(2, 0, 1) * 255).astype(
                    np.uint8
                )
                t = (ICEFIRE_CMAP(gt[i])[..., :3].transpose(2, 0, 1) * 255).astype(
                    np.uint8
                )
                d = (ICEFIRE_CMAP(diff[i])[..., :3].transpose(2, 0, 1) * 255).astype(
                    np.uint8
                )
                frames[i, :, :, :width] = g
                frames[i, :, :, width : 2 * width] = t
                frames[i, :, :, 2 * width :] = d

            return wandb.Video(frames, fps=fps, format="mp4")

        for ch_idx in range(min(num_channels, len(channel_titles))):
            title = channel_titles[ch_idx]
            if title is None:
                continue
            videos[f"generated/comparison_{title}"] = _make_video(
                generated[:, ch_idx], target[:, ch_idx]
            )

        if num_channels >= 2:
            gen_vort = np.stack(
                [
                    _compute_vorticity(generated[i, 0], generated[i, 1])
                    for i in range(num_frames)
                ]
            )
            tgt_vort = np.stack(
                [
                    _compute_vorticity(target[i, 0], target[i, 1])
                    for i in range(num_frames)
                ]
            )
            videos["generated/comparison_vorticity"] = _make_video(gen_vort, tgt_vort)

        if videos:
            wandb.log(videos)

    except Exception as e:
        logger.error(f"Error logging comparison video: {e}")


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
                label="±1 std",
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

        # Log metrics
        if scalar_metrics:
            wandb.log(scalar_metrics)

    except Exception as e:
        logger.error(f"Failed to log temporal deviation metrics: {e}")


def log_multi_temporal_deviation(
    metrics_per_model: Dict[str, Dict[str, any]],
    prefix: str = "test",
    references: List[Dict[str, Any]] = None,
):
    """Log temporal deviation plots for multiple models on a single figure.

    Parameters
    ----------
    metrics_per_model: Mapping from model name to calculated metrics.
    prefix: Log prefix for wandb.
    references: Optional list of mappings specifying mean/std values of
        external results to plot as dashed curves. Each entry should contain a
        ``name`` field and metric dictionaries like ``mse_per_step`` with
        a scalar ``mean`` and optional ``std``.
    """
    try:
        if not metrics_per_model:
            return

        def plot_metric(metric_key: str, ylabel: str):
            color_cycle = itertools.cycle(
                plt.rcParams["axes.prop_cycle"].by_key()["color"]
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            num_steps = None
            for color, (name, metrics) in zip(color_cycle, metrics_per_model.items()):
                sims = [
                    sim[metric_key]
                    for sim in metrics.get("simulations", [])
                    if metric_key in sim
                ]
                if not sims:
                    continue
                arr = np.asarray(sims)
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                if num_steps is None:
                    num_steps = len(mean)
                steps = np.arange(len(mean))
                ax.plot(steps, mean, label=name, color=color, linewidth=2.5)
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

            if references and num_steps:
                for ref in references:
                    ref_name = ref.get("name", "reference")
                    metric = ref.get(metric_key)
                    if not metric or "mean" not in metric:
                        continue
                    ref_mean = float(metric["mean"])
                    ref_steps = np.arange(num_steps)
                    color = next(color_cycle)
                    ax.plot(
                        ref_steps,
                        np.full(num_steps, ref_mean),
                        label=ref_name,
                        linestyle="--",
                        linewidth=2,
                        color=color,
                    )
                    if "std" in metric and metric["std"] is not None:
                        ref_std = float(metric["std"])
                        ax.fill_between(
                            ref_steps,
                            ref_mean - ref_std,
                            ref_mean + ref_std,
                            color=color,
                            alpha=0.2,
                        )

            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_facecolor("white")
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.margins(x=0.02, y=0.05)
            ax.legend(frameon=False, fontsize=10)
            plt.tight_layout()
            return wandb.Image(fig_to_image(fig))

        images = {}
        mse_img = plot_metric("mse_per_step", "Mean Squared Error")
        if mse_img:
            images[f"{prefix}/mse_over_time"] = mse_img

        rmse_img = plot_metric("rmse_per_step", "Root Mean Squared Error")
        if rmse_img:
            images[f"{prefix}/rmse_over_time"] = rmse_img

        if images:
            wandb.log(images)
    except Exception as e:
        logger.error(f"Failed to log multi-model temporal deviation metrics: {e}")
