"""Tests for the shared selector pieces training and inference both read through.

Everything here guards a failure that is SILENT: the selector still runs, still writes a
mask, and only the BF1 at the end is wrong. That is how the first version of this code
shipped with the prediction-space conversion in apply_contour_selector.py and not in
train_contour_selector.py.

  - the two backends of the top-share rule disagreeing, so the validation precision
    stops predicting what the inference run will keep;
  - the disparity conversion being skipped, which hands the selector an inverted signal
    (disparity rises as depth falls);
  - `in_channels` disagreeing with what `build_input` stacks, which only shows up as a
    shape error if you are lucky and as a wrong channel order if you are not.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contour_selector import (PRED_SPACES, UNet, in_channels, keep_count, standardise,
                              to_log_depth, top_share_mask, top_share_mask_torch)


class TopShareTest(unittest.TestCase):
    def test_keeps_the_requested_share(self):
        rng = np.random.default_rng(0)
        score = rng.normal(size=(64, 64))
        cand = rng.random((64, 64)) < 0.3
        keep = top_share_mask(score, cand, 0.1)
        self.assertAlmostEqual(keep.sum() / cand.sum(), 0.1, delta=0.01)

    def test_keeps_only_candidates(self):
        rng = np.random.default_rng(1)
        score = rng.normal(size=(32, 32))
        cand = rng.random((32, 32)) < 0.2
        self.assertTrue((top_share_mask(score, cand, 0.5) & ~cand).sum() == 0)

    def test_keeps_the_highest_scoring_candidates(self):
        score = np.arange(100, dtype=float).reshape(10, 10)
        cand = np.ones((10, 10), bool)
        keep = top_share_mask(score, cand, 0.1)
        self.assertEqual(sorted(score[keep].tolist()), list(range(90, 100)))

    def test_empty_candidate_set_is_empty_not_an_error(self):
        keep = top_share_mask(np.zeros((8, 8)), np.zeros((8, 8), bool), 0.5)
        self.assertEqual(keep.sum(), 0)
        self.assertEqual(keep.dtype, np.bool_)

    def test_at_least_one_pixel_survives_rounding(self):
        cand = np.zeros((40, 40), bool)
        cand[0, :3] = True                      # 3 candidates, 1% of which rounds to 0
        self.assertEqual(top_share_mask(np.arange(1600.).reshape(40, 40), cand, 0.01).sum(), 1)
        self.assertEqual(keep_count(3, 0.01), 1)

    def test_the_numpy_and_torch_backends_agree(self):
        """The validation metric must predict what the inference run keeps."""
        rng = np.random.default_rng(2)
        for retention in (0.02, 0.06, 0.25, 0.5):
            score = rng.normal(size=(48, 64))
            cand = rng.random((48, 64)) < 0.4
            a = top_share_mask(score, cand, retention)
            b = top_share_mask_torch(torch.from_numpy(score),
                                     torch.from_numpy(cand), retention).numpy()
            self.assertTrue(np.array_equal(a, b),
                            f"backends disagree at retention {retention}")


class PredSpaceTest(unittest.TestCase):
    def test_log_space_is_passed_through(self):
        pred = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(to_log_depth(pred, "log"), pred))

    def test_disparity_is_inverted_into_log_depth(self):
        """Disparity rises as depth falls, so the ORDER must flip."""
        disp = np.array([[4.0, 2.0, 1.0]], dtype=np.float32)      # near -> far
        out = to_log_depth(disp, "disparity")
        self.assertLess(out[0, 0], out[0, 1])
        self.assertLess(out[0, 1], out[0, 2])

    def test_disparity_conversion_is_log_depth_up_to_an_affine(self):
        depth = np.array([[1.0, 2.0, 4.0, 8.0]], dtype=np.float32)
        out = to_log_depth(1.0 / depth, "disparity")
        np.testing.assert_allclose(out, np.log(depth), rtol=1e-5)

    def test_zero_disparity_does_not_blow_up(self):
        out = to_log_depth(np.zeros((2, 2), np.float32), "disparity")
        self.assertTrue(np.isfinite(out).all())

    def test_an_unknown_space_is_rejected(self):
        with self.assertRaises(ValueError):
            to_log_depth(np.zeros((2, 2), np.float32), "inverse")

    def test_the_declared_spaces_all_work(self):
        for space in PRED_SPACES:
            self.assertEqual(to_log_depth(np.ones((3, 3), np.float32), space).shape, (3, 3))


class StandardiseTest(unittest.TestCase):
    def test_zero_mean_unit_variance(self):
        rng = np.random.default_rng(3)
        out = standardise(rng.normal(5.0, 3.0, size=(32, 32)).astype(np.float32))
        self.assertAlmostEqual(float(out.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out.std()), 1.0, places=4)

    def test_it_removes_the_affine_the_prediction_is_invariant_to(self):
        rng = np.random.default_rng(4)
        pred = rng.normal(size=(24, 24)).astype(np.float32)
        np.testing.assert_allclose(standardise(pred), standardise(3.0 * pred + 7.0),
                                   atol=1e-4)

    def test_a_constant_frame_does_not_divide_by_zero(self):
        self.assertTrue(np.isfinite(standardise(np.full((8, 8), 2.0, np.float32))).all())


class ChannelWidthTest(unittest.TestCase):
    def test_widths_match_the_ablations(self):
        self.assertEqual(in_channels(), 5)
        self.assertEqual(in_channels(no_rgb=True), 2)
        self.assertEqual(in_channels(no_sam=True), 4)
        self.assertEqual(in_channels(no_rgb=True, no_sam=True), 1)

    def test_the_network_accepts_every_declared_width(self):
        for kw in ({}, {"no_rgb": True}, {"no_sam": True}, {"no_rgb": True, "no_sam": True}):
            c = in_channels(**kw)
            net = UNet(c, base=8).eval()
            with torch.no_grad():
                out = net(torch.zeros(1, c, 32, 32))
            self.assertEqual(tuple(out.shape), (1, 1, 32, 32), f"failed for {kw}")

    def test_the_released_selector_has_the_published_size(self):
        """Two counts, because they differ and both have been quoted.

        1,928,993 is what `model.parameters()` sums and what the training log rounds to
        1.93M. The checkpoint holds 1,931,823 tensor elements, the extra 2,830 being the
        BatchNorm running statistics, which are buffers and not trained.
        """
        net = UNet(5, 32)
        self.assertEqual(sum(p.numel() for p in net.parameters()), 1_928_993)
        self.assertEqual(sum(b.numel() for b in net.buffers()), 2_830)


if __name__ == "__main__":
    unittest.main()
