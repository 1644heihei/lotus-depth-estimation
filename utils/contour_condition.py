"""Contour maps as an extra UNet input, for the contour-conditioning experiment.

docs/contour_sharpening_findings.md measured a contour to be worth +279.9% BF1 net of
control when used as a barrier in post-processing - the largest ceiling anywhere in this
investigation, with no training and abs_rel improving. It is unreachable that way because
the post-hoc operation trusts the contour's position and needs it within 1.71px, while SAM
delivers 2.00px.

Training is the thing that could bridge those 0.29px, because a model can learn to read a
contour as "a step is near here" rather than "the step is exactly here". That is the one
capability post-processing cannot have.

Two facts make the input-channel route viable, both measured rather than assumed. A contour
survives Lotus's VAE almost intact - 91.1% of its pixels come back exactly, 99.8% within
the 1.71px budget - so the 8x downsample does not destroy the very thing the channel exists
to carry. And expand_unet_conv_in already zero-initialises the appended kernel slices, so
training starts at exactly the pretrained model rather than paying the tax up front that
every LoRA run in this repo has paid.

The caches store the union of the instance masks; the contour is derived here so the
band width stays defined in one place.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def contour_of(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """The mask's boundary, `width` pixels thick."""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    u = mask.astype(np.uint8)
    return cv2.dilate(u, k).astype(bool) & ~cv2.erode(u, k).astype(bool)


class ContourCache:
    """Union masks on disk, turned into contour maps in the training range."""

    def __init__(self, root: str | Path, width: int = 1):
        self.root = Path(root)
        self.width = width
        self._cache: dict[str, np.ndarray] = {}

    def _path(self, image_path, data_root: Path) -> Path:
        """Locate the cache entry, tolerating a data_root that differs by a split level.

        build_sam_masks.py is pointed at <hypersim>/train while training is pointed at
        <hypersim>, so the relative paths differ by one leading directory. Rather than
        regenerate 59k masks, both spellings are tried - and a missing file returns an
        empty contour silently, which is the failure this experiment could not detect from
        its own results.
        """
        img = Path(image_path).resolve()
        try:
            rel = img.relative_to(Path(data_root).resolve())
        except ValueError:
            rel = Path(img.name)
        cand = self.root / rel.parent / f"{rel.stem}_seg.npz"
        if cand.is_file() or len(rel.parts) < 2:
            return cand
        return self.root / Path(*rel.parts[1:]).parent / f"{rel.stem}_seg.npz"

    def check_root(self, sample_paths, data_root, shape, min_hit_frac: float = 0.2) -> None:
        """Refuse to start when the mask cache does not resolve.

        A missing file yields an empty contour, so a mis-rooted cache trains for hours on
        blank channels and reports that contour conditioning does nothing - a failure
        invisible in the loss. The equivalent guard on class-name prompts caught exactly
        this, on a path that differed by one directory.
        """
        paths = [str(p) for p in sample_paths]
        if not paths:
            return
        # Content, not existence: build_sam_masks writes a file with n=0 for frames
        # without detections, so checking is_file() passes even when every contour is
        # empty. Hypersim's own rate is ~60% non-empty.
        hits = 0
        for p in paths:
            q = self._path(p, Path(data_root))
            if q.is_file() and int(np.load(q)["n"]) > 0:
                hits += 1
        if hits / len(paths) < min_hit_frac:
            raise RuntimeError(
                f"Contour mask root {self.root} gave {hits}/{len(paths)} frames with any "
                f"mask. Expected roughly 60%. Probed e.g. "
                f"{self._path(paths[0], Path(data_root))}"
            )

    def contour_for(self, image_path, data_root, shape) -> np.ndarray:
        """[H,W] float32 in {-1, +1}: +1 on the contour, -1 elsewhere.

        The same [-1,1] range the RGB and depth inputs use, so the VAE sees the kind of
        signal it was trained on rather than a 0/1 map with an unfamiliar mean.

        The mask is stored at the frame's own resolution while training runs at a resized
        one, so it is decoded at its native size and the CONTOUR is resized afterwards.
        Decoding straight into the requested shape reads the wrong number of bits and
        reinterprets them at the wrong width - which yields an empty map rather than an
        error, and cost a smoke run to find.
        """
        key = (str(image_path), shape)
        if key in self._cache:
            return self._cache[key]
        h, w = shape
        p = self._path(image_path, data_root)
        out = np.full((h, w), -1.0, np.float32)
        if p.is_file():
            d = np.load(p)
            n = int(d["n"])
            if n > 0:
                bits = d["packed"].shape[-1] * 8
                nh, nw = self._native_shape(image_path, bits)
                m = np.unpackbits(d["packed"], axis=-1, count=nh * nw).reshape(n, nh, nw)
                c = contour_of(np.any(m.astype(bool), axis=0), self.width)
                if (nh, nw) != (h, w):
                    c = cv2.resize(c.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                out[c] = 1.0
        self._cache[key] = out
        return out

    def _native_shape(self, image_path, bits: int) -> tuple[int, int]:
        """The resolution the mask was packed at, read from the frame itself."""
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
        if h * w == bits:
            return h, w
        raise ValueError(
            f"mask holds {bits} bits but {Path(image_path).name} is {h}x{w}={h*w}"
        )

    def batch(self, image_paths, data_root, batch_size, shape, device, dtype, shuffle=None):
        """[B,1,H,W] for the batch, padded with empty maps beyond len(image_paths).

        `shuffle` supplies another image's paths for the control run: the same amount of
        contour, drawn from a different scene. Any gain that survives it belongs to the
        contour describing THIS image rather than to the channel being non-empty.
        """
        src = shuffle if shuffle is not None else image_paths
        h, w = shape
        arr = np.full((batch_size, 1, h, w), -1.0, np.float32)
        for i, p in enumerate(src[:batch_size]):
            arr[i, 0] = self.contour_for(p, data_root, shape)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
