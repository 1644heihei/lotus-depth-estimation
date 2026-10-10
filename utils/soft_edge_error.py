"""Soft Edge Error, transcribed from Marigold V2's own implementation.

The second boundary metric this work reports. BF1 asks whether a depth step is detected
where ground truth has one; SEE asks how wrong the depth VALUE is at a step, forgiving a
small displacement. They can disagree, which is the point of reporting both: an
improvement visible in only one of them is an improvement in that metric's definition,
not necessarily in the depth map.

Transcribed rather than imported. The implementation lives in another repository with its
own virtualenv, and a cross-repo import would break the moment either moves; transcribing
it also forces the details to be read rather than trusted. Those details are not
guessable from the paper's one-line description:

  - SEE is averaged over GT BOUNDARY PIXELS ONLY, not over the image. Boundaries come
    from 4-neighbour jumps in inverse depth above 0.05, which is the SMD-Nets definition
    (arXiv:2104.03866) adapted to depth.
  - The error is an ABSOLUTE difference in depth units, so unlike BF1 it is not
    scale-invariant and the prediction must be aligned to GT before scoring.
  - Invalid GT inside the patch is excluded from the minimum rather than counted as zero.

Lower is better. `tests/test_soft_edge_error.py` checks this transcription against values
computed from the original source.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SEE_PATCH_SIZES = (3, 5, 7)


def depth_boundary_mask(depth: torch.Tensor, valid: torch.Tensor,
                        disparity_jump_threshold: float = 0.05,
                        eps: float = 1e-6) -> torch.Tensor:
    """GT pixels adjacent to an inverse-depth jump, in either direction."""
    vm = valid.bool()
    if not torch.any(vm):
        return torch.zeros_like(vm)

    disp = torch.zeros_like(depth, dtype=torch.float32)
    safe = vm & (depth > eps) & torch.isfinite(depth)
    if not torch.any(safe):
        return torch.zeros_like(vm)
    disp[safe] = 1.0 / torch.clamp(depth[safe], min=eps)

    boundary = torch.zeros_like(vm)
    thr = float(disparity_jump_threshold)

    edge = vm[:, :-1] & vm[:, 1:] & (torch.abs(disp[:, :-1] - disp[:, 1:]) > thr)
    boundary[:, :-1] |= edge
    boundary[:, 1:] |= edge

    edge = vm[:-1, :] & vm[1:, :] & (torch.abs(disp[:-1, :] - disp[1:, :]) > thr)
    boundary[:-1, :] |= edge
    boundary[1:, :] |= edge

    return boundary & vm


def soft_edge_error(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                    patch_size: int = 5, disparity_jump_threshold: float = 0.05,
                    eps: float = 1e-6) -> float:
    """SEE_k: mean over GT boundary pixels of the best match within a k x k patch.

    `pred` must already be aligned to `gt`: the error is absolute, so an affine-invariant
    prediction scored unaligned measures the alignment, not the boundaries.
    """
    if patch_size % 2 == 0 or patch_size < 1:
        raise ValueError(f"patch_size must be odd and >= 1, got {patch_size}")

    vm = (valid.bool() & torch.isfinite(pred) & torch.isfinite(gt)
          & (pred > eps) & (gt > eps))
    if not torch.any(vm):
        return float("nan")

    boundary = depth_boundary_mask(gt, vm, disparity_jump_threshold, eps)
    score_mask = boundary & vm
    if not torch.any(score_mask):
        return float("nan")

    radius = patch_size // 2
    gt_patches = F.unfold(gt[None, None].float(), kernel_size=patch_size,
                          padding=radius)[0]
    vm_patches = F.unfold(vm[None, None].float(), kernel_size=patch_size,
                          padding=radius)[0] > 0.5

    abs_err = torch.abs(gt_patches - pred.reshape(1, -1).float())
    abs_err = torch.where(vm_patches, abs_err, torch.full_like(abs_err, float("inf")))
    min_local_err = abs_err.min(dim=0).values.reshape_as(pred)
    return float(min_local_err[score_mask].mean())


def soft_edge_errors(pred, gt, valid, patch_sizes=SEE_PATCH_SIZES,
                     disparity_jump_threshold: float = 0.05) -> dict:
    """SEE at several patch sizes, from numpy or torch arrays."""
    to_t = lambda a: a if isinstance(a, torch.Tensor) else torch.from_numpy(a)
    p, g, v = to_t(pred).float(), to_t(gt).float(), to_t(valid)
    return {f"see_{k}": soft_edge_error(p, g, v, k, disparity_jump_threshold)
            for k in patch_sizes}
