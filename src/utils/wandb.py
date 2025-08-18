import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from PIL import Image
import seaborn as sns
import pytorch_lightning as pl
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from utils.log import get_logger
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ICEFIRE_CMAP = sns.color_palette("icefire", as_cmap=True)
logger = get_logger("utils.wandb")


def _compute_vorticity(vy: np.ndarray, vx: np.ndarray) -> np.ndarray:
    """Compute 2D vorticity from velocity components."""
    vy_dy, vy_dx = np.gradient(vy)
    vx_dy, vx_dx = np.gradient(vx)
    return vx_dy - vy_dx


def fig_to_image(fig) -> PIL.Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    return PIL.Image.open(buffer)


def _create_video_frames(data: np.ndarray, fps: int = 10, with_colorbar: bool = False, scale: float = 4.0, normalize: str = "global") -> wandb.Video:
    """Create video frames from field data.
    
    data: 3D array [T, H, W]
    """
    assert data.ndim == 3, f"Expected [T, H, W], got {data.shape}"
        
    x = np.nan_to_num(data).astype(np.float32, copy=False)
    T, H, W = x.shape

    def _norm(arr, mode):
        if mode == "frame":
            mins = arr.min(axis=(1, 2))
            maxs = arr.max(axis=(1, 2))
            rng = np.maximum(maxs - mins, 1e-12)
            n = np.clip((arr - mins[:, None, None]) / rng[:, None, None], 0, 1)
            return n, (mins[0], maxs[0]), (mins, maxs)
        vmin, vmax = float(arr.min()), float(arr.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-12
        n = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        return n, (vmin, vmax), None

    if not with_colorbar:
        n, _, _ = _norm(x, normalize)
        frames = (ICEFIRE_CMAP(n)[..., :3] * 255).astype(np.uint8).transpose(0, 3, 1, 2)
        return wandb.Video(frames, fps=fps, format="mp4")

    n, (v0, V0), fm = _norm(x, normalize)
    ow, oh = int(round(W * scale)), int(round(H * scale))
    fig = plt.figure(figsize=(ow / 100, oh / 100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(x[0], cmap=ICEFIRE_CMAP, norm=mcolors.Normalize(
        vmin=v0, vmax=V0, clip=True), interpolation="nearest")
    cax = inset_axes(ax, width="5%", height="100%", loc="upper right", borderpad=0)
    cb = plt.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.set_yticks([])
    tmax = cb.ax.text(0.5, 0.98, f"{V0:.3g}", ha="center", va="top",
                      transform=cb.ax.transAxes, color="k", fontsize=10, rotation=90)
    tmin = cb.ax.text(0.5, 0.02, f"{v0:.3g}", ha="center", va="bottom",
                      transform=cb.ax.transAxes, color="k", fontsize=10, rotation=90)

    frames = np.empty((T, oh, ow, 3), dtype=np.uint8)
    for t in range(T):
        im.set_data(x[t])
        if fm is not None:
            im.set_clim(fm[0][t], fm[1][t])
            tmax.set_text(f"{fm[1][t]:.3g}")
            tmin.set_text(f"{fm[0][t]:.3g}")
        canvas.draw()
        frames[t] = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(oh, ow, 3)

    plt.close(fig)
    return wandb.Video(frames.transpose(0, 3, 1, 2), fps=fps, format="mp4")


def _create_video_frames_comparison(gen: np.ndarray, gt: np.ndarray, fps: int = 10) -> wandb.Video:
    """Side-by-side comparison of generated, ground-truth, and difference frames.
    
    gen: Generated frames [T, H, W]
    gt: Ground truth frames [T, H, W]
    """
    vals = np.nan_to_num(np.stack([gen, gt, gen - gt], axis=0)).astype(np.float32, copy=False)
    vmin, vmax = vals.min(), vals.max()

    if vmax <= vmin:
        norm = np.zeros_like(vals, dtype=np.float32)
    else:
        norm = (vals - vmin) / (vmax - vmin)

    gen_n, gt_n, diff_n = norm

    def to_rgb(x):
        return ICEFIRE_CMAP(x)[..., :3]  # [T, H, W, 3] in [0,1]

    side_by_side = np.concatenate([to_rgb(gen_n), to_rgb(gt_n), to_rgb(diff_n)], axis=2)  # [T, H, 3W, 3]
    frames = np.moveaxis((side_by_side * 255).astype(np.uint8), -1, 1)  # [T, 3, H, 3W]

    return wandb.Video(frames, fps=fps, format="mp4")


def log_sample(
    data: np.ndarray,
    module: pl.LightningModule,
    channel_titles: List[str],
    prefix: str = "model"
) -> None:
    """Log image samples to wandb.
    
    data: Data array [C, H, W]
    """
    try:
        assert len(data.shape) == 3, f"Expected 3D array [C, H, W], got {data.shape}"
            
        C, H, W = data.shape
        
        assert len(channel_titles) == C, f"Length of channel_titles must match number of channels"

        images_to_log = {}

        for idx in range(C):
            title = channel_titles[idx]
            if title is None:
                continue

            field_data = np.nan_to_num(data[idx])
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(field_data, cmap=ICEFIRE_CMAP, aspect="equal")
            plt.colorbar(im, ax=ax)

            images_to_log[f"{prefix}/frame_{title}"] = wandb.Image(fig_to_image(fig))
            plt.close(fig)

        for key, img in images_to_log.items():
            module.logger.experiment.log({key: img})

    except Exception as e:
        logger.error(f"Error logging samples: {e}")


def log_samples_video(
    data: np.ndarray,
    module: pl.LightningModule,
    channel_titles: List[str],
    fps: int = 10,
    prefix: str = "model",
    with_colorbar: bool = True,
    normalize: str = "global"
) -> None:
    """Log video samples to wandb.

    data: Data array [T, C, H, W]
    """
    try:
        assert len(data.shape) == 4, f"Expected 4D array [T, C, H, W], got {data.shape}"
            
        T, C, H, W = data.shape

        videos_to_log = {}

        for idx in range(C):
            title = channel_titles[idx]
            if title is None:
                continue

            videos_to_log[f"{prefix}/video_{title}"] = _create_video_frames(
                data[:, idx], fps, with_colorbar, normalize=normalize
            )

        for key, video in videos_to_log.items():
            module.logger.experiment.log({key: video})

    except Exception as e:
        logger.error(f"Error logging video samples: {e}")


def log_comparison_video(
    gen: np.ndarray,
    gt: np.ndarray,
    channel_titles: List[str],
    module: pl.LightningModule,
    fps: int = 10,
    prefix: str = "model"
) -> None:
    """Log side-by-side comparison videos to wandb.
    
    gen: Generated data [T, C, H, W]
    gt: Ground truth data [T, C, H, W]
    """
    try:
        assert len(gen.shape) == 4, f"Expected 4D array [T, C, H, W], got {gen.shape}"
        assert gen.shape == gt.shape, "Generated and ground truth shapes must match"
            
        T, C, H, W = gen.shape

        videos_to_log = {}

        for idx in range(C):
            title = channel_titles[idx]
            if title is None:
                continue

            videos_to_log[f"{prefix}/comparison_{title}"] = _create_video_frames_comparison(
                gen[:, idx], gt[:, idx], fps=fps
            )

        for key, video in videos_to_log.items():
            module.logger.experiment.log({key: video})

    except Exception as e:
        logger.error(f"Error logging comparison video: {e}")


def log_mse(
    metrics: Dict[str, Any],
    module: pl.LightningModule,
    prefix: str = "test"
) -> None:
    """Log MSE metrics to wandb."""
    try:
        assert "mse_mean" in metrics and "mse_time" in metrics, "Missing required metrics"
            
        module.log(f"{prefix}/mse_mean", metrics["mse_mean"], on_epoch=True, sync_dist=True)

        arr = np.array(metrics["mse_time"])
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
        ax.set_ylabel("Mean Squared Error", fontsize=12, fontweight="normal")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_facecolor("white")
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.margins(x=0.02, y=0.05)
        ax.legend(frameon=False, fontsize=10)
        plt.tight_layout()

        image = wandb.Image(fig_to_image(fig))
        plt.close(fig)

        module.logger.experiment.log({
            f"{prefix}/mse_time": image
        })

    except Exception as e:
        logger.error(f"Failed to log MSE: {e}")


def log_denoising_video(
    cond: torch.Tensor,
    channel_titles: List[str],
    module: pl.LightningModule,
    fps: int = 1,
    frames: int = 50,
    normalize: str = "global",
    prefix: str = "debugging"
) -> None:
    """Log denoising process video to wandb."""
    try:
        _, denoising_steps = module.generate_samples(
            cond=cond,
            num_steps=min(module.timesteps, frames),
            debug=True
        )

        denoising_sequence = denoising_steps.numpy()

        videos_to_log = {}
        for channel_idx, title in enumerate(channel_titles):
            if title is None or channel_idx >= denoising_sequence.shape[2]:
                continue

            channel_data = denoising_sequence[:, 0, channel_idx]
            videos_to_log[f"{prefix}/denoising_{title}"] = _create_video_frames(
                channel_data, fps=fps, with_colorbar=True, normalize=normalize
            )

        for key, video in videos_to_log.items():
            module.logger.experiment.log({key: video})

    except Exception as e:
        logger.error(f"Error logging denoising visualization: {e}")
