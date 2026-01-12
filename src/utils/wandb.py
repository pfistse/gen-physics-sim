import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
import pytorch_lightning as pl
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import utils.plotting as plt_utils

ICEFIRE_CMAP = sns.color_palette("icefire", as_cmap=True)


def get_plot_dir(module: pl.LightningModule) -> str:
    return os.path.join(os.getcwd(), f"plots/epoch_{module.current_epoch}")


def fig_to_image(fig) -> PIL.Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    return PIL.Image.open(buffer)


def _create_video_frames(
    data: np.ndarray,
    fps: int = 10,
    with_colorbar: bool = False,
    scale: float = 4.0,
    normalize: str = "global",
) -> wandb.Video:
    x = np.nan_to_num(data).astype(np.float32, copy=False)
    T, H, W = x.shape

    if normalize == "frame":
        mins = x.min(axis=(1, 2))
        maxs = x.max(axis=(1, 2))
        vmins, vmaxs = mins, maxs
    else:
        gmin, gmax = float(x.min()), float(x.max())
        if not np.isfinite(gmin) or not np.isfinite(gmax):
            gmin, gmax = 0.0, 1.0
        if gmax <= gmin:
            gmax = gmin + 1e-12
        vmins, vmaxs = np.array([gmin] * T), np.array([gmax] * T)

    fig, ax = plt.subplots(
        figsize=(W / 100 * scale, H / 100 * scale), dpi=100, layout="constrained"
    )
    canvas = FigureCanvas(fig)
    im = ax.imshow(x[0], cmap=ICEFIRE_CMAP, vmin=vmins[0], vmax=vmaxs[0])
    cbar = fig.colorbar(im, ax=ax) if with_colorbar else None

    frames = []
    for t in range(T):
        im.set_data(x[t])
        im.set_clim(vmins[t], vmaxs[t])
        if cbar:
            cbar.update_normal(im)
        canvas.draw()
        frames.append(
            np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
                canvas.get_width_height()[1], canvas.get_width_height()[0], 3
            )
        )

    plt.close(fig)
    return wandb.Video(np.stack(frames).transpose(0, 3, 1, 2), fps=fps, format="mp4")


def _create_video_frames_comparison(
    gen: np.ndarray, gt: np.ndarray, fps: int = 10, scale: float = 4.0
) -> wandb.Video:
    gen_x = np.nan_to_num(gen).astype(np.float32, copy=False)
    gt_x = np.nan_to_num(gt).astype(np.float32, copy=False)
    diff_x = gen_x - gt_x
    vals = np.stack([gen_x, gt_x, diff_x], axis=0)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-12

    T, H, W = gen_x.shape
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(3 * W / 100 * scale, H / 100 * scale),
        dpi=100,
        layout="constrained",
    )
    canvas = FigureCanvas(fig)
    ims = [
        ax.imshow(d[0], cmap=ICEFIRE_CMAP, vmin=vmin, vmax=vmax)
        for ax, d in zip(axes, [gen_x, gt_x, diff_x])
    ]
    for ax, t in zip(axes, ["Generated", "Ground Truth", "Difference"]):
        ax.set_title(t)
    fig.colorbar(ims[0], ax=axes, orientation="vertical")

    frames = []
    for t in range(T):
        for im, d in zip(ims, [gen_x, gt_x, diff_x]):
            im.set_data(d[t])
        canvas.draw()
        frames.append(
            np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
                canvas.get_width_height()[1], canvas.get_width_height()[0], 3
            )
        )

    plt.close(fig)
    return wandb.Video(np.stack(frames).transpose(0, 3, 1, 2), fps=fps, format="mp4")


def log_sample(
    data: np.ndarray,
    module: pl.LightningModule,
    channel_titles: List[str],
    prefix: str = "model",
    pretitle: str = "sample",
    cmap: Optional[Union[str, mcolors.Colormap]] = ICEFIRE_CMAP,
) -> None:
    C, H, W = data.shape
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config

    with plt_utils.style_context(plot_cfg):
        for idx, title in enumerate(channel_titles):
            if title is None:
                continue
            field_data = np.nan_to_num(data[idx])
            fig = plt_utils.plot_field(field_data, cmap=cmap)
            name = f"{prefix}_{pretitle}_{title}".replace("/", "_").replace(" ", "_")
            plt_utils.save_figure(
                fig, {"data": field_data}, plot_dir, name, formats=plot_cfg.plot_formats
            )
            module.logger.experiment.log(
                {f"{prefix}/{pretitle}_{title}": wandb.Image(fig_to_image(fig))}
            )
            plt.close(fig)


def log_samples_video(
    data: np.ndarray,
    module: pl.LightningModule,
    channel_titles: List[str],
    cfg: Any,
    prefix: str = "model",
    pre_title: str = "video",
) -> None:
    videos = {}
    for idx, title in enumerate(channel_titles):
        if title is None:
            continue
        videos[f"{prefix}/{pre_title}_{title}"] = _create_video_frames(
            data[:, idx], cfg.fps, normalize=cfg.normalize
        )
    module.logger.experiment.log(videos)


def log_comparison_video(
    gen: np.ndarray,
    gt: np.ndarray,
    channel_titles: List[str],
    module: pl.LightningModule,
    cfg: Any,
    prefix: str = "model",
) -> None:
    videos = {}
    for idx, title in enumerate(channel_titles):
        if title is None:
            continue
        videos[f"{prefix}/comparison_{title}"] = _create_video_frames_comparison(
            gen[:, idx], gt[:, idx], fps=cfg.fps
        )
    module.logger.experiment.log(videos)


def log_mse(
    metrics: Dict[str, Any], module: pl.LightningModule, cfg: Any, prefix: str = "test"
) -> None:
    module.log(f"{prefix}/mse_mean", metrics["mse_mean"], on_step=True, on_epoch=False)
    arr = np.array(metrics["mse_time"])
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config

    with plt_utils.style_context(plot_cfg):
        fig = plt_utils.plot_mean_std(
            np.mean(arr, 0)[None],
            np.std(arr, 0)[None],
            np.arange(1, len(arr[0]) + 1),
            "Time Step",
            "Mean Squared Error",
            ["steelblue"],
        )
        name = f"{prefix}_mse_time".replace("/", "_")
        plt_utils.save_figure(
            fig, {"mse_time": arr}, plot_dir, name, formats=plot_cfg.plot_formats
        )
        module.logger.experiment.log(
            {f"{prefix}/mse_time": wandb.Image(fig_to_image(fig))}
        )
        plt.close(fig)


def log_acc_metrics_steps(
    means: List[List[float]],
    stds: List[List[float]],
    steps: List[int],
    module: pl.LightningModule,
    prefix: str = "debug",
    title: str = "std_steps",
) -> None:
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config
    means_np, stds_np = np.asarray(means), np.asarray(stds)

    with plt_utils.style_context(plot_cfg):
        fig = plt_utils.plot_mean_std(
            means_np,
            stds_np,
            np.asarray(steps),
            "Step",
            "Relative L2 error",
            ["darkorange", "steelblue"],
        )
        name = f"{prefix}_{title}".replace("/", "_")
        plt_utils.save_figure(
            fig,
            {"means": means_np, "stds": stds_np, "steps": steps},
            plot_dir,
            name,
            formats=plot_cfg.plot_formats,
        )
        image = wandb.Image(fig_to_image(fig))
        plt.close(fig)

    table = wandb.Table(columns=["step", "series_id", "mean", "std"])
    for i, (series_m, series_s) in enumerate(zip(means, stds)):
        for step, m, s in zip(steps, series_m, series_s):
            table.add_data(int(step), i, float(m), float(s))
    module.logger.experiment.log(
        {f"{prefix}/{title}": image, f"raw/{prefix}/{title}_data": table}
    )


def log_std_rel_err_grid(
    mean: List[List[float]],
    std: List[List[float]],
    steps: List[int],
    g_coefs: List[float],
    module: pl.LightningModule,
    prefix: str = "debug",
    title: str = "std_rel_err_grid",
) -> None:
    mean_arr, std_arr = np.asarray(mean, dtype=np.float32), np.asarray(
        std, dtype=np.float32
    )
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config

    with plt_utils.style_context(plot_cfg):
        line_fig = plt_utils.plot_std_rel_err_lines(
            mean_arr, std_arr, np.asarray(steps), g_coefs
        )
        heatmap_fig = plt_utils.plot_std_rel_err_heatmap(
            mean_arr, np.asarray(steps), g_coefs
        )
        name = f"{prefix}_{title}".replace("/", "_")
        plt_utils.save_figure(
            line_fig,
            {"mean": mean_arr, "std": std_arr},
            plot_dir,
            f"{name}_line",
            formats=plot_cfg.plot_formats,
        )
        plt_utils.save_figure(
            heatmap_fig,
            None,
            plot_dir,
            f"{name}_heatmap",
            formats=plot_cfg.plot_formats,
        )
        line_img, heatmap_img = wandb.Image(fig_to_image(line_fig)), wandb.Image(
            fig_to_image(heatmap_fig)
        )
        plt.close(line_fig)
        plt.close(heatmap_fig)

    table = wandb.Table(columns=["step", "g_coef", "mean", "std"])
    for step_idx, step in enumerate(steps):
        for g_idx, g in enumerate(g_coefs):
            table.add_data(
                int(step),
                float(g),
                float(mean_arr[g_idx, step_idx]),
                float(std_arr[g_idx, step_idx]),
            )
    module.logger.experiment.log(
        {
            f"{prefix}/{title}_line": line_img,
            f"{prefix}/{title}_heatmap": heatmap_img,
            f"raw/{prefix}/{title}_data": table,
        }
    )


def log_std_grid(
    std_steps: List[np.ndarray],
    mean_steps: List[np.ndarray],
    steps: List[Union[int, str]],
    module: pl.LightningModule,
    channel_titles: List[str],
    prefix: str = "metric",
    pretitle: str = "std_steps",
    cmap: Optional[Union[str, mcolors.Colormap]] = "viridis",
) -> None:
    arr0 = std_steps[0]
    C, H, W = arr0.shape
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config

    with plt_utils.style_context(plot_cfg):
        for c in range(C):
            title = channel_titles[c] if c < len(channel_titles) else f"ch{c}"
            if title is None:
                continue

            vmin = min(float(arr[c].min()) for arr in std_steps)
            vmax = max(float(arr[c].max()) for arr in std_steps)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0

            images = [step_data[c] for step_data in std_steps]
            titles = [
                f"s={s}\nstd={float(np.nanmean(img)):.3g}"
                for s, img in zip(steps, images)
            ]
            fig = plt_utils.plot_grid_images(
                images, titles, ncols=4, cmap=cmap, vmin=vmin, vmax=vmax, cbar_title=""
            )
            name = f"{prefix}_{pretitle}_{title}".replace("/", "_").replace(" ", "_")
            plt_utils.save_figure(
                fig,
                {"std_steps": np.stack(images, axis=0), "steps": np.array(steps)},
                plot_dir,
                name,
                formats=plot_cfg.plot_formats,
            )
            module.logger.experiment.log(
                {f"{prefix}/{pretitle}_{title}": wandb.Image(fig_to_image(fig))}
            )
            plt.close(fig)


def log_field_comparison(
    arr_model: np.ndarray,
    arr_solver: np.ndarray,
    module: pl.LightningModule,
    channel_titles: List[str],
    cfg: Any,
    cmap: str,
    prefix: str,
    pretitle: str,
) -> None:
    arr_model, arr_solver = np.asarray(arr_model, dtype=np.float32), np.asarray(
        arr_solver, dtype=np.float32
    )
    C, _, _ = arr_model.shape
    cmap = ICEFIRE_CMAP if cmap.lower() == "icefire" else plt.get_cmap(cmap)
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config

    with plt_utils.style_context(plot_cfg):
        for c in range(C):
            title = channel_titles[c] if c < len(channel_titles) else f"ch{c}"
            if title is None:
                continue
            gen_x, gt_x = np.nan_to_num(arr_model[c]), np.nan_to_num(arr_solver[c])
            images, titles = [gen_x, gt_x], ["Model", "Solver"]
            if cfg.diff:
                images.append(gen_x - gt_x)
                titles.append("Difference")
            fig = plt_utils.plot_comparison_row(images, titles, cmap=cmap)
            name = f"{prefix}_{pretitle}_{title}".replace("/", "_").replace(" ", "_")
            plt_utils.save_figure(
                fig,
                {"model": gen_x, "solver": gt_x},
                plot_dir,
                name,
                formats=plot_cfg.plot_formats,
            )
            module.logger.experiment.log(
                {f"{prefix}/{pretitle}_{title}": wandb.Image(fig_to_image(fig))}
            )
            plt.close(fig)


def log_mean_std_grid(
    entries: List[Dict[str, Any]],
    module: pl.LightningModule,
    channel_titles: List[str],
    cfg: Any,
    prefix: str,
) -> None:
    if not entries:
        return
    _log_seed_grid(
        entries,
        module,
        channel_titles,
        cfg,
        prefix,
        "mean_comparison_grid",
        cfg.cmap_mean,
        "_mean_model_arr",
        "_mean_solver_arr",
        int(cfg.grid_cols),
    )
    _log_seed_grid(
        entries,
        module,
        channel_titles,
        cfg,
        prefix,
        "std_comparison_grid",
        cfg.cmap_std,
        "_std_model_arr",
        "_std_solver_arr",
        int(cfg.grid_cols),
    )


def _log_seed_grid(
    entries: List[Dict[str, Any]],
    module: pl.LightningModule,
    channel_titles: List[str],
    cfg: Any,
    prefix: str,
    pretitle: str,
    cmap_name: str,
    model_key: str,
    solver_key: str,
    grid_cols: int,
) -> None:
    cmap = ICEFIRE_CMAP if cmap_name.lower() == "icefire" else plt.get_cmap(cmap_name)
    C = entries[0][model_key].shape[0]
    titles = channel_titles or []
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config

    with plt_utils.style_context(plot_cfg):
        for c in range(C):
            title = titles[c] if c < len(titles) else f"ch{c}"
            if title is None:
                continue
            fig = plt_utils.plot_seed_grid(
                entries,
                model_key,
                solver_key,
                c,
                grid_cols,
                cfg.diff,
                cmap,
                cfg.show_overshoot,
                float(cfg.overshoot_tresh),
            )
            name = f"{prefix}_{pretitle}_{title}".replace("/", "_").replace(" ", "_")
            data_to_save = {
                "seeds": np.array([e["seed"] for e in entries]),
                "model": np.stack([e[model_key][c] for e in entries]),
                "solver": np.stack([e[solver_key][c] for e in entries]),
            }
            plt_utils.save_figure(
                fig, data_to_save, plot_dir, name, formats=plot_cfg.plot_formats
            )
            module.logger.experiment.log(
                {f"{prefix}/{pretitle}_{title}": wandb.Image(fig_to_image(fig))}
            )
            plt.close(fig)


def log_enstrophy(
    metrics: Dict[str, Any],
    module: pl.LightningModule,
    prefix: str = "metric",
    title: str = "enstrophy_spectrum",
    cfg: Any = None,
) -> None:
    sm, ss = np.asarray(metrics["_spec_model"], dtype=np.float32), np.asarray(
        metrics["_spec_solver"], dtype=np.float32
    )
    if not cfg.incl_const:
        sm, ss = sm[1:], ss[1:]
    plot_dir, plot_cfg = get_plot_dir(module), module.eval_config

    with plt_utils.style_context(plot_cfg):
        fig = plt_utils.plot_enstrophy_spectrum(
            np.arange(1, len(sm) + 1), sm, ss, cfg.loglog
        )
        name = f"{prefix}_{title}".replace("/", "_")
        plt_utils.save_figure(
            fig,
            {"spec_model": sm, "spec_solver": ss},
            plot_dir,
            name,
            formats=plot_cfg.plot_formats,
        )
        module.logger.experiment.log(
            {f"{prefix}/{title}": wandb.Image(fig_to_image(fig))}
        )
        plt.close(fig)


def log_denoising_video(
    cond: torch.Tensor,
    channel_titles: List[str],
    module: pl.LightningModule,
    cfg: Any,
    prefix: str = "debugging",
) -> None:
    _, denoising_steps = module.generate_samples(
        cond=cond, num_steps=module.timesteps, debug=True
    )
    denoising_sequence = denoising_steps.numpy()
    videos = {}
    for i, title in enumerate(channel_titles):
        if title is None or i >= denoising_sequence.shape[2]:
            continue
        videos[f"{prefix}/denoising_{title}"] = _create_video_frames(
            denoising_sequence[:, 0, i],
            fps=cfg.fps,
            with_colorbar=True,
            normalize=cfg.normalize,
        )
    module.logger.experiment.log(videos)
