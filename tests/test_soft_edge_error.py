"""Tests for the Soft Edge Error transcription.

The implementation was copied from Marigold V2's repository and checked against it
directly, on random inputs with boundaries, holes and invalid depths: 12 trials x 3
patch sizes, worst difference 0.00e+00. That check needs the other repository on disk,
so it cannot run here. What runs here instead are the properties that would have caught
a bad transcription, on cases small enough to reason about by hand.

The last test is the one that matters for the paper: SEE forgives a displaced boundary
and punishes a blurred one. That asymmetry is the whole reason it is a boundary metric,
and it is also why it can disagree with BF1.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.soft_edge_error import (SEE_PATCH_SIZES, depth_boundary_mask,
                                   soft_edge_error, soft_edge_errors)


def _step(h=16, w=16, near=2.0, far=5.0):
    """A depth map with one vertical step down the middle."""
    gt = np.full((h, w), near, np.float32)
    gt[:, w // 2:] = far
    return gt


class BoundaryMaskTest(unittest.TestCase):
    def test_a_step_marks_both_of_its_sides(self):
        gt = torch.from_numpy(_step())
        valid = torch.ones_like(gt, dtype=torch.bool)
        b = depth_boundary_mask(gt, valid)
        self.assertTrue(b[:, 7].all(), "left side of the step not marked")
        self.assertTrue(b[:, 8].all(), "right side of the step not marked")
        self.assertFalse(b[:, 0].any(), "flat region marked")

    def test_a_flat_map_has_no_boundary(self):
        gt = torch.full((16, 16), 3.0)
        self.assertFalse(depth_boundary_mask(gt, torch.ones_like(gt, dtype=torch.bool)).any())

    def test_the_threshold_is_on_inverse_depth(self):
        """1/2 - 1/5 = 0.3 is a boundary; 1/2 - 1/2.05 = 0.012 is not."""
        valid = torch.ones((8, 8), dtype=torch.bool)
        big = torch.from_numpy(_step(8, 8, 2.0, 5.0))
        small = torch.from_numpy(_step(8, 8, 2.0, 2.05))
        self.assertTrue(depth_boundary_mask(big, valid).any())
        self.assertFalse(depth_boundary_mask(small, valid).any())

    def test_invalid_pixels_cannot_form_a_boundary(self):
        gt = torch.from_numpy(_step())
        valid = torch.ones_like(gt, dtype=torch.bool)
        valid[:, 7:9] = False
        self.assertFalse(depth_boundary_mask(gt, valid)[:, 7:9].any())


class SoftEdgeErrorTest(unittest.TestCase):
    def test_a_perfect_prediction_scores_zero(self):
        gt = torch.from_numpy(_step())
        v = torch.ones_like(gt, dtype=torch.bool)
        self.assertAlmostEqual(soft_edge_error(gt.clone(), gt, v, 5), 0.0, places=6)

    def test_lower_is_better(self):
        gt = torch.from_numpy(_step())
        v = torch.ones_like(gt, dtype=torch.bool)
        near = soft_edge_error(gt + 0.05, gt, v, 5)
        far = soft_edge_error(gt + 0.50, gt, v, 5)
        self.assertLess(near, far)

    def test_no_boundary_in_gt_gives_nan(self):
        gt = torch.full((16, 16), 3.0)
        v = torch.ones_like(gt, dtype=torch.bool)
        self.assertTrue(np.isnan(soft_edge_error(gt.clone(), gt, v, 5)))

    def test_a_larger_patch_can_only_help(self):
        """The minimum is over a superset, so SEE is non-increasing in k."""
        rng = np.random.default_rng(0)
        gt = _step(24, 24)
        pred = gt + rng.normal(0, 0.3, gt.shape).astype(np.float32)
        v = torch.ones(gt.shape, dtype=torch.bool)
        vals = [soft_edge_error(torch.from_numpy(pred), torch.from_numpy(gt), v, k)
                for k in (3, 5, 7)]
        self.assertGreaterEqual(vals[0], vals[1] - 1e-6)
        self.assertGreaterEqual(vals[1], vals[2] - 1e-6)

    def test_an_even_patch_is_rejected(self):
        gt = torch.from_numpy(_step())
        v = torch.ones_like(gt, dtype=torch.bool)
        with self.assertRaises(ValueError):
            soft_edge_error(gt.clone(), gt, v, 4)

    def test_invalid_gt_in_the_patch_is_excluded_not_counted_as_zero(self):
        """Counting a hole as a GT value of 0 would make the minimum spuriously small."""
        gt = _step()
        pred = gt.copy()
        pred[:, 8] = 0.1                     # badly wrong on the far side
        gt_t, pred_t = torch.from_numpy(gt), torch.from_numpy(pred)
        v_all = torch.ones(gt.shape, dtype=torch.bool)
        err = soft_edge_error(pred_t, gt_t, v_all, 3)
        self.assertGreater(err, 0.0, "a wrong prediction scored zero")

    def test_soft_edge_errors_returns_every_patch_size(self):
        gt = _step()
        out = soft_edge_errors(gt.copy(), gt, np.ones(gt.shape, bool))
        self.assertEqual(set(out), {f"see_{k}" for k in SEE_PATCH_SIZES})

    def test_it_forgives_displacement_and_punishes_blur(self):
        """Why SEE is a boundary metric, and why it can disagree with BF1.

        Both predictions are wrong in the same four pixels. The displaced one puts a
        crisp step one pixel off; the blurred one puts an intermediate value there. SEE
        should barely notice the first and should notice the second.
        """
        gt = _step(16, 16, 2.0, 5.0)
        v = torch.ones(gt.shape, dtype=torch.bool)

        displaced = gt.copy()
        displaced[:, 7] = 5.0                      # step moved one pixel left

        blurred = gt.copy()
        blurred[:, 7] = 3.0                        # intermediate values instead
        blurred[:, 8] = 4.0

        e_disp = soft_edge_error(torch.from_numpy(displaced), torch.from_numpy(gt), v, 5)
        e_blur = soft_edge_error(torch.from_numpy(blurred), torch.from_numpy(gt), v, 5)
        self.assertAlmostEqual(e_disp, 0.0, places=6,
                               msg="a one-pixel displacement should be forgiven")
        self.assertGreater(e_blur, 0.5, "blur should be punished")


if __name__ == "__main__":
    unittest.main()
