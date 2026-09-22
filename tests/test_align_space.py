"""Tests for the two alignment spaces.

Two things can go wrong silently here, and both would look like a bad model
rather than a bad aligner:

  - the log aligner fits in the wrong space, or drops the negative half of an
    affine-invariant log-depth prediction the way the disparity aligner does;
  - the disparity path drifts from `align_to_gt` in eval_object_oracle_ceiling.py,
    which every published number in docs/ was produced with.

So the log space is checked by round trip - build a prediction that IS an affine
transform of log GT, and require the aligner to return GT - and the disparity
path is checked against the original function on the same inputs.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.align_space import (  # noqa: E402
    DEPTH_MAX,
    align_to_gt_disparity,
    align_to_gt_log,
    depth2log_space,
    get_aligner,
    log_space2depth,
)


def synthetic_gt(h=64, w=80, lo=0.6, hi=8.0, seed=0):
    """A smooth depth field in metres, well inside the NYUv2 range."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = lo + (hi - lo) * (0.5 + 0.5 * np.sin(3 * xx / w) * np.cos(2 * yy / h))
    return base + rng.normal(0, 0.01, base.shape)


class TestLogSpaceRoundTrip(unittest.TestCase):
    def test_conversions_invert(self):
        gt = synthetic_gt()
        log, mask = depth2log_space(gt)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(log_space2depth(log), gt, rtol=1e-9, atol=1e-9)

    def test_recovers_gt_from_affine_log_prediction(self):
        # This is what Log-stage2 emits: log depth up to an unknown scale and shift.
        gt = synthetic_gt()
        valid = np.ones_like(gt, bool)
        log_gt, _ = depth2log_space(gt)
        for a, b in [(0.5, -0.3), (2.0, 1.7), (0.31, 0.0), (1.0, -1.0)]:
            pred = a * log_gt + b
            out = align_to_gt_log(pred, gt, valid)
            self.assertIsNotNone(out, f"aligner returned None for a={a} b={b}")
            np.testing.assert_allclose(out, gt, rtol=1e-4, atol=1e-4)

    def test_negative_predictions_are_not_discarded(self):
        # Log-stage2's output spans [-1, 1]; the disparity aligner's `pred > 0`
        # filter would throw away everything below the affine zero crossing.
        gt = synthetic_gt()
        valid = np.ones_like(gt, bool)
        log_gt, _ = depth2log_space(gt)
        pred = (log_gt - log_gt.mean()) / (np.abs(log_gt - log_gt.mean()).max())
        self.assertLess(pred.min(), 0.0, "test fixture should contain negatives")
        self.assertGreater((pred < 0).mean(), 0.2, "and a substantial share of them")
        out = align_to_gt_log(pred, gt, valid)
        np.testing.assert_allclose(out, gt, rtol=1e-4, atol=1e-4)

    def test_disparity_aligner_is_materially_worse_on_log_input(self):
        # Feeding log depth to the disparity aligner is the mistake this module
        # exists to prevent, and it does not announce itself: no exception, and
        # the error lands at a few percent - close enough to Marigold V2's
        # published 3.6% AbsRel that an absolute gate could not tell the two
        # aligners apart. So assert the RELATIVE property instead: the correct
        # aligner must beat the wrong one by orders of magnitude on input it owns.
        gt = synthetic_gt()
        valid = np.ones_like(gt, bool)
        log_gt, _ = depth2log_space(gt)
        pred = (log_gt - log_gt.mean()) / (np.abs(log_gt - log_gt.mean()).max())

        right = align_to_gt_log(pred, gt, valid)
        wrong = align_to_gt_disparity(pred, gt, valid)
        self.assertIsNotNone(right)
        self.assertIsNotNone(wrong, "the wrong aligner fails silently, not loudly")

        err_right = float((np.abs(right - gt) / gt).mean())
        err_wrong = float((np.abs(wrong - gt) / gt).mean())
        self.assertLess(err_right, 1e-4, f"log aligner should be near-exact, got {err_right}")
        self.assertGreater(err_wrong, 100 * err_right,
                           f"disparity aligner ({err_wrong}) should be far worse than "
                           f"the log aligner ({err_right}) on log input")


class TestDisparitySpaceUnchanged(unittest.TestCase):
    def test_matches_the_original_align_to_gt(self):
        from eval_object_oracle_ceiling import align_to_gt as original

        rng = np.random.default_rng(7)
        gt = synthetic_gt(seed=3)
        valid = np.ones_like(gt, bool)
        valid[:4, :] = False  # an Eigen-crop-like hole
        pred = (1.0 / gt) * 0.8 + 0.05 + rng.normal(0, 1e-3, gt.shape)
        a = original(pred, gt, valid)
        b = align_to_gt_disparity(pred, gt, valid)
        self.assertIsNotNone(a)
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)

    def test_recovers_gt_from_affine_disparity_prediction(self):
        gt = synthetic_gt(seed=5)
        valid = np.ones_like(gt, bool)
        pred = 0.7 * (1.0 / gt) + 0.2
        out = align_to_gt_disparity(pred, gt, valid)
        np.testing.assert_allclose(out, gt, rtol=1e-4, atol=1e-4)


class TestSelector(unittest.TestCase):
    def test_returns_the_right_callable(self):
        self.assertIs(get_aligner("disparity"), align_to_gt_disparity)
        self.assertIs(get_aligner("log"), align_to_gt_log)

    def test_unknown_space_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            get_aligner("depth")

    def test_too_few_valid_pixels_returns_none(self):
        gt = synthetic_gt()
        valid = np.zeros_like(gt, bool)
        valid[0, :10] = True
        log_gt, _ = depth2log_space(gt)
        self.assertIsNone(align_to_gt_log(log_gt, gt, valid))
        self.assertIsNone(align_to_gt_disparity(1.0 / gt, gt, valid))

    def test_output_is_clipped_to_the_nyuv2_range(self):
        gt = synthetic_gt()
        valid = np.ones_like(gt, bool)
        log_gt, _ = depth2log_space(gt)
        out = align_to_gt_log(log_gt * 3.0, gt, valid)  # a badly scaled prediction
        self.assertLessEqual(out.max(), DEPTH_MAX + 1e-9)
        self.assertGreater(out.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
