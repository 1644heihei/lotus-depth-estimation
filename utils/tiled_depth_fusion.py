"""Tiled high-resolution depth inference: tile geometry, affine alignment, fusion.

Lotus resizes its input so the longest edge is `processing_res`, then the VAE
downsamples 8x - so the input resolution sets the latent budget for the whole scene.
Phase 0 measured the consequence: 512 -> 768 gains 9.8% abs_rel with no training,
but 1024 loses 14.4% because the model leaves its training distribution.

Tiling buys latent capacity without leaving that distribution: run the model at a
resolution it handles well, but on a crop covering fewer source pixels. The latent
density multiplier is (global source max edge) / (tile source max edge).

Each tile's prediction is affine-invariant disparity with its own arbitrary scale and
shift, so tiles must be fitted to the global pass before they can be blended.

See docs/tiled_inference_plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def slice(self):
        return (slice(self.y1, self.y2), slice(self.x1, self.x2))

    @property
    def shape(self):
        return (self.y2 - self.y1, self.x2 - self.x1)


def make_tiles(h: int, w: int, grid: int = 2, overlap: float = 0.25) -> List[Tile]:
    """Split an h x w image into a grid x grid cover with fractional overlap.

    Tiles are clamped to the image, and the last row/column is anchored to the far edge
    so the cover is exact even when the stride does not divide evenly.
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if grid == 1:
        return [Tile(0, 0, w, h)]

    def spans(total: int) -> List[Tuple[int, int]]:
        # size * (grid - (grid-1)*overlap) == total
        size = int(round(total / (grid - (grid - 1) * overlap)))
        size = min(max(size, 1), total)
        stride = (total - size) / (grid - 1)
        out = []
        for i in range(grid):
            a = int(round(i * stride))
            a = min(a, total - size)
            out.append((a, a + size))
        return out

    ys, xs = spans(h), spans(w)
    return [Tile(x1, y1, x2, y2) for (y1, y2) in ys for (x1, x2) in xs]


def latent_density_multiplier(h: int, w: int, tiles: Sequence[Tile]) -> float:
    """How much more latent capacity per source pixel a tile gets vs the global pass.

    Both passes run at the same processing_res, so the ratio is just how much less
    of the image each tile has to cover (compared on the longest edge, which is what
    resize_max_res keys on).
    """
    global_edge = max(h, w)
    tile_edge = max(max(t.shape) for t in tiles)
    return global_edge / max(tile_edge, 1)


def feather_weight(tile: Tile, h: int, w: int, ramp_px: int = 32) -> np.ndarray:
    """Blend weight for one tile: 1 inside, ramping to 0 at seams.

    Edges that coincide with the image border are NOT ramped - nothing overlaps there,
    so fading out would leave the border underweighted.
    """
    th, tw = tile.shape
    ramp = max(1, min(ramp_px, th // 2, tw // 2))

    def axis(n: int, fade_lo: bool, fade_hi: bool) -> np.ndarray:
        a = np.ones(n, dtype=np.float32)
        # cosine ramp: smooth first derivative at both ends, unlike a linear ramp
        r = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp + 2)[1:-1]))
        if fade_lo:
            a[:ramp] = np.minimum(a[:ramp], r)
        if fade_hi:
            a[-ramp:] = np.minimum(a[-ramp:], r[::-1])
        return a

    wy = axis(th, tile.y1 > 0, tile.y2 < h)
    wx = axis(tw, tile.x1 > 0, tile.x2 < w)
    out = np.zeros((h, w), dtype=np.float32)
    out[tile.slice] = wy[:, None] * wx[None, :]
    return out


def fit_affine(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    trim: float = 0.25,
) -> Tuple[float, float]:
    """Least-squares a, b minimizing |a*src + b - dst| over mask, with outlier trimming.

    The tile and the global pass genuinely disagree where the tile resolves detail the
    global pass missed; a plain fit lets those pixels drag the coefficients. Refitting
    on the best-fitting (1 - trim) fraction keeps the fit on the agreeing majority.
    Falls back to shift-only if the fit is degenerate or flips the sign.
    """
    x = src[mask].astype(np.float64)
    y = dst[mask].astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 16:
        return 1.0, float(np.median(y - x)) if x.size else 0.0

    def lstsq(xx, yy):
        A = np.stack([xx, np.ones_like(xx)], axis=1)
        return np.linalg.lstsq(A, yy, rcond=None)[0]

    a, b = lstsq(x, y)
    if 0.0 < trim < 1.0:
        resid = np.abs(a * x + b - y)
        keep = resid <= np.quantile(resid, 1.0 - trim)
        if keep.sum() >= 16 and np.var(x[keep]) > 1e-12:
            a, b = lstsq(x[keep], y[keep])

    if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-8:
        return 1.0, float(np.median(y - x))
    return float(a), float(b)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float32)
    import cv2

    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigma, borderType=cv2.BORDER_REPLICATE)


def fuse_average(
    global_disp: np.ndarray,
    tiles: Sequence[Tile],
    aligned_crops: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
) -> np.ndarray:
    """F1: feathered weighted average of the aligned tiles.

    Simple, but a tile that gets the local structure wrong writes that error straight
    into the output. Used as the control for fuse_detail_transfer.
    """
    num = np.zeros_like(global_disp, dtype=np.float32)
    den = np.zeros_like(global_disp, dtype=np.float32)
    for tile, crop, wgt in zip(tiles, aligned_crops, weights):
        num[tile.slice] += wgt[tile.slice] * crop.astype(np.float32)
        den += wgt
    out = global_disp.astype(np.float32).copy()
    covered = den > 1e-6
    out[covered] = num[covered] / den[covered]
    return out


def fuse_detail_transfer(
    global_disp: np.ndarray,
    tiles: Sequence[Tile],
    aligned_crops: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    sigma: float = 8.0,
) -> np.ndarray:
    """F2: keep the global pass's low frequencies, take only high frequencies from tiles.

    A tile sees a crop, so it has better detail but worse context - it can get the
    overall layout of its crop wrong. Taking only the detail band means such a mistake
    cannot corrupt the scene structure, which the global pass still owns.

    Formulated on the tile-minus-global difference rather than on the tile itself. Two
    reasons: the low band being discarded is exactly the structural disagreement we do
    not trust, and a tile identical to the global contributes exactly zero, so fusion
    reduces to the global pass instead of leaking blur artifacts at the seams.
    """
    num = np.zeros_like(global_disp, dtype=np.float32)
    den = np.zeros_like(global_disp, dtype=np.float32)
    for tile, crop, wgt in zip(tiles, aligned_crops, weights):
        diff = crop.astype(np.float32) - global_disp[tile.slice].astype(np.float32)
        detail = diff - _gaussian_blur(diff, sigma)
        num[tile.slice] += wgt[tile.slice] * detail
        den += wgt
    out = global_disp.astype(np.float32).copy()
    covered = den > 1e-6
    out[covered] += num[covered] / den[covered]
    return out


def align_tile_to_global(
    tile: Tile,
    tile_disp: np.ndarray,
    global_disp: np.ndarray,
    *,
    ramp_px: int = 32,
    trim: float = 0.25,
) -> np.ndarray:
    """Fit one tile's disparity onto the global pass; returns the aligned tile-sized crop.

    The fit uses the tile's interior only: the feathered border contributes little to
    the output but is where tile predictions are least reliable, so letting it into the
    fit would bias the coefficients.
    """
    th, tw = tile.shape
    inset = max(0, min(ramp_px, th // 4, tw // 4))
    interior = np.zeros((th, tw), dtype=bool)
    interior[inset : th - inset or th, inset : tw - inset or tw] = True

    g_local = global_disp[tile.slice]
    finite = np.isfinite(tile_disp) & np.isfinite(g_local)
    a, b = fit_affine(tile_disp, g_local, interior & finite, trim=trim)
    return (a * tile_disp.astype(np.float32) + b).astype(np.float32)
