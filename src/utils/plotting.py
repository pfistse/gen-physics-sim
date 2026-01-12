import os
import math
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
from typing import Optional, List, Union, Tuple, Dict, Any

# -----------------------------------------------------------------------------
# Configuration & Utils
# -----------------------------------------------------------------------------

@contextmanager
def style_context(cfg: Any):
    """
    Context manager to apply plotting styles (fonts, latex, etc.) from config.
    Expects cfg to have attributes like:
      - usetex (bool)
      - fontsize (int)
      - figsize (tuple) - handled per plot usually, but defaults could be here
      - dpi (int)
    """
    usetex = cfg.usetex
    fontsize = cfg.fontsize

    params = {
        "text.usetex": usetex,
        "font.family": "serif" if usetex else "sans-serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        # "figure.dpi": getattr(cfg, "dpi", 100), # Let caller control DPI or default
    }

    with plt.rc_context(params):
        yield

def save_figure(fig: plt.Figure, data: Optional[Dict[str, np.ndarray]], path: str, name: str, formats: List[str] = ["pdf"]):
    """
    Saves the figure and optionally raw data.

    Args:
        fig: The matplotlib figure.
        data: Dictionary of numpy arrays to save (for reproducibility).
        path: Directory to save in.
        name: Base filename (without extension).
        formats: List of extensions to save (e.g., ["pdf", "pgf", "png"]).
    """
    os.makedirs(path, exist_ok=True)
    base_path = os.path.join(path, name)

    # Save figure
    for fmt in formats:
        try:
            fig.savefig(f"{base_path}.{fmt}", bbox_inches="tight", pad_inches=0.05)
        except Exception as e:
            print(f"Error saving {fmt} to {base_path}: {e}")

    # Save data
    if data is not None:
        try:
            np.savez_compressed(f"{base_path}.npz", **data)
        except Exception as e:
            print(f"Error saving data to {base_path}: {e}")

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------

def plot_mean_std(
    means: np.ndarray,
    stds: np.ndarray,
    x: np.ndarray,
    xlabel: str,
    ylabel: str,
    colors: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plots mean lines with standard deviation shading.
    means, stds: [num_lines, num_steps]
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    num_lines = means.shape[0]
    for i in range(num_lines):
        c = colors[i] if colors and i < len(colors) else None
        ax.fill_between(x, means[i] - stds[i], means[i] + stds[i], alpha=0.2, color=c, zorder=1)
        ax.plot(x, means[i], color=c, linewidth=2.5, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, axis="both", linewidth=0.6, alpha=0.35)

    return fig

def plot_std_rel_err_lines(
    mean: np.ndarray,
    std: Optional[np.ndarray],
    steps: np.ndarray,
    g_coefs: List[float],
) -> plt.Figure:
    """
    Line plot for std relative error grid.
    mean, std: [len(g_coefs), len(steps)]
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, g_coef in enumerate(g_coefs):
        mu = mean[idx]
        sigma = std[idx] if std is not None else None
        label = f"g={g_coef:.3g}"

        if sigma is not None:
            ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.15, zorder=1)
        ax.plot(steps, mu, linewidth=2.0, label=label, zorder=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Relative L2 error")
    ax.set_xticks(steps)
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)

    return fig

def plot_std_rel_err_heatmap(
    mean: np.ndarray,
    steps: np.ndarray,
    g_coefs: List[float],
) -> plt.Figure:
    """Heatmap for std relative error grid."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean, aspect="auto", origin="lower", cmap="magma")

    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(np.arange(len(g_coefs)))
    ax.set_yticklabels([f"{g:.3g}" for g in g_coefs])

    ax.set_xlabel("Step")
    ax.set_ylabel("g_follmer")
    ax.set_title("Relative L2 error")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig

def plot_grid_images(
    images: List[np.ndarray],
    titles: List[str],
    ncols: int = 4,
    cmap: Any = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_title: Optional[str] = None
) -> plt.Figure:
    """
    Generic grid plotter for 2D fields.
    images: List of [H, W] arrays.
    """
    n_plots = len(images)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        constrained_layout=False
    )

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = np.array([axes])

    im = None
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes_flat[i]
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused
    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    # Colorbar
    # If we have a grid, put colorbar on the right
    if n_plots > 0:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        if cbar_title:
            cbar_ax.set_title(cbar_title)

    return fig

def plot_field(
    data: np.ndarray,
    title: str = "",
    cmap: Any = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True
) -> plt.Figure:
    """Plots a single 2D field."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    if colorbar:
        plt.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    return fig

def plot_comparison_row(
    images: List[np.ndarray],
    titles: List[str],
    cmap: Any = "viridis",
    diff: bool = False
) -> plt.Figure:
    """
    Plots a single row comparison (e.g. Model, Solver, [Diff]).
    """
    ncols = len(images)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    vals = np.concatenate([img.flatten() for img in images])
    vmin = vals.min() if vals.size > 0 else 0
    vmax = vals.max() if vals.size > 0 else 1

    ims = []
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    fig.colorbar(ims[0], ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    return fig

def plot_seed_grid(
    entries: List[Dict[str, Any]],
    model_key: str,
    solver_key: str,
    channel_idx: int,
    grid_cols: int = 3,
    diff: bool = False,
    cmap: Any = "viridis",
    show_overshoot: bool = False,
    overshoot_tresh: float = 0.0,
) -> plt.Figure:
    """
    Plots a grid of comparisons across multiple seeds (entries).
    """
    num_seeds = len(entries)
    n_panels = 3 if diff else 2

    num_groups = math.ceil(num_seeds / grid_cols)
    fig, axes = plt.subplots(
        num_groups,
        grid_cols * n_panels,
        figsize=(3.0 * grid_cols * n_panels, 3.0 * num_groups),
        dpi=320,
    )
    if num_groups == 1:
        axes = np.expand_dims(axes, axis=0)
    if grid_cols * n_panels == 1:
        axes = np.expand_dims(axes, axis=-1)

    axes = np.asarray(axes)
    axes = axes.reshape(num_groups, grid_cols * n_panels)

    for idx, entry in enumerate(entries):
        row = idx // grid_cols
        col_base = (idx % grid_cols) * n_panels

        gen_x = np.nan_to_num(entry[model_key][channel_idx]).astype(np.float32, copy=False)
        gt_x = np.nan_to_num(entry[solver_key][channel_idx]).astype(np.float32, copy=False)
        panels = [gen_x, gt_x]
        panel_titles = ["Model", "Solver"]
        if diff:
            panels.append(gen_x - gt_x)
            panel_titles.append("Difference")

        local_vals = np.stack(panels, axis=0)
        vmin = float(np.min(local_vals))
        vmax = float(np.max(local_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0

        for p_idx, panel in enumerate(panels):
            ax = axes[row, col_base + p_idx]
            ax.imshow(
                panel, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(panel_titles[p_idx])
            if p_idx == 0:
                label = f"seed {entry['seed']}"
                if show_overshoot:
                    denom = float(np.mean(np.abs(gt_x)))
                    if denom < 1e-12:
                        denom = 1e-12
                    diff_val = float(np.mean(gen_x - gt_x))
                    coeff = diff_val / denom
                    coeff_color = None
                    if abs(coeff) >= overshoot_tresh:
                        coeff_color = "green" if coeff > 0 else "red"
                    coeff_text = f"{coeff:+.3f}"
                    label = f"{label}\novershoot: {coeff_text}"

                if show_overshoot and coeff_color is not None:
                    ax.set_ylabel(label, color=coeff_color)
                else:
                    ax.set_ylabel(label)

    # Hide unused
    for idx in range(num_seeds, num_groups * grid_cols):
        row = idx // grid_cols
        col_base = (idx % grid_cols) * n_panels
        for p in range(n_panels):
            axes[row, col_base + p].set_visible(False)

    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.1)
    return fig

def plot_enstrophy_spectrum(
    k: np.ndarray,
    spec_model: np.ndarray,
    spec_solver: np.ndarray,
    loglog: bool = True
) -> plt.Figure:
    """Plots enstrophy spectrum."""
    fig, ax = plt.subplots(figsize=(10, 6))

    eps = 1e-12
    spec_model = np.maximum(spec_model, eps)
    spec_solver = np.maximum(spec_solver, eps)

    ax.plot(k, spec_solver, label="Solver", color="tab:blue", linewidth=2.0)
    ax.plot(k, spec_model, label="Model", color="tab:orange", linewidth=2.0, linestyle="--")

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Enstrophy")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False)

    return fig
