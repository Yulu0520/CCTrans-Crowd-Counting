"""Count-preserving adaptive Gaussian targets for density regression."""

import numpy as np
from scipy.spatial import cKDTree


def gen_gaussian_map(height, width, points, downsample_ratio=8,
                     beta=0.3, fallback_sigma=15.0):
    """Render at image resolution, then sum blocks to preserve head counts.

    Sigma is beta times the mean distance to up to three other heads.
    A singleton uses fallback_sigma; all sigmas are at least one image pixel.
    Each kernel is truncated at four sigmas and normalized after boundary clipping.
    """
    if height % downsample_ratio or width % downsample_ratio:
        raise ValueError("Density map dimensions must be divisible by downsample_ratio")
    if beta <= 0 or fallback_sigma <= 0:
        raise ValueError("Gaussian beta and fallback sigma must be positive")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    density = np.zeros((height, width), dtype=np.float32)
    if len(points):
        if not np.isfinite(points).all():
            raise ValueError("Head coordinates must be finite")
        if len(points) == 1:
            sigmas = np.array([fallback_sigma])
        else:
            distances, _ = cKDTree(points).query(points, k=min(4, len(points)))
            sigmas = beta * distances[:, 1:].mean(axis=1)
        sigmas = np.maximum(sigmas, 1.0)
        centers = np.rint(points).astype(int)
        centers[:, 0] = np.clip(centers[:, 0], 0, width - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, height - 1)
        for (x, y), sigma in zip(centers, sigmas):
            radius = int(np.ceil(4 * sigma))
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            xx = np.arange(x0, x1) - x
            yy = np.arange(y0, y1) - y
            kernel = np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2 * sigma ** 2))
            density[y0:y1, x0:x1] += (kernel / kernel.sum()).astype(np.float32)
    ratio = downsample_ratio
    return density.reshape(height // ratio, ratio, width // ratio, ratio).sum(axis=(1, 3))
