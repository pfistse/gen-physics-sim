from __future__ import annotations

import math
from pathlib import Path

import imageio
import cv2
import numpy as np
import seaborn as sns

from src.physics.inc_stoc_vort import simulate

# Configuration paths
ROOT_DIR = Path(__file__).parent.parent
OUT_PATH = ROOT_DIR / "data" / "inc_stoc_vort.npz"
VIDEO_PATH = ROOT_DIR / "data" / "inc_stoc_vort_preview.mp4"
PREVIEW_FRAMES = 20


def _render_preview(data: np.ndarray, path: Path, frames: int, scale_factor: int = 4) -> None:
    cmap = sns.color_palette("icefire", as_cmap=True)
    seq = data[:frames, 0]
    vmin, vmax = seq.min(), seq.max()

    imgs = []
    for field in seq:
        norm = (field - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(field)
        rgb = cmap(norm)[..., :3]
        img_uint8 = (rgb * 255).astype(np.uint8)

        h, w = img_uint8.shape[:2]
        img_resized = cv2.resize(
            img_uint8,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_NEAREST
        )

        imgs.append(img_resized)

    imageio.mimsave(path, imgs, fps=2, quality=9)


def main():
    data = simulate()
    data = data.astype(np.float32)
    data = np.expand_dims(data, axis=0)

    np.savez_compressed(
        OUT_PATH,
        vorticity=data
    )

    _render_preview(data[0], VIDEO_PATH, min(PREVIEW_FRAMES, len(data[0])))

    print(f"Saved numpy array with shape {data.shape} to {OUT_PATH} and preview to {VIDEO_PATH}")


if __name__ == "__main__":
    main()
