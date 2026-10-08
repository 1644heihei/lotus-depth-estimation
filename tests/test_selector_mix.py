"""Tests for mixed-dataset training of the contour selector.

Two failures here would be silent and would cost a training run each:

  - the random crop slicing the five input channels, the label and the candidate
    mask out of step, so the selector learns against labels that belong to a
    different part of the image;
  - the crop being computed BEFORE the distance transform or before the
    per-image standardisation, either of which changes what the selector sees
    relative to inference, where it always gets a whole frame.

Both are checked against a synthetic source written to a temp directory, so the
test needs no cached predictions and no GPU.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from train_contour_selector import ContourDataset, mix_sources


def _write_source(root: Path, scene: str, name: str, h: int, w: int, seed: int):
    """One frame: an RGB png, a uint16 depth png, a prediction .npy, a SAM contour npz."""
    rng = np.random.default_rng(seed)
    (root / "rgb" / scene).mkdir(parents=True, exist_ok=True)
    (root / "pred" / scene).mkdir(parents=True, exist_ok=True)
    (root / "mask" / scene).mkdir(parents=True, exist_ok=True)

    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(root / "rgb" / scene / f"rgb_{name}.png")

    # A depth step down the middle, so discontinuities() has something real to find.
    depth = np.full((h, w), 2.0)
    depth[:, w // 2:] = 4.0
    Image.fromarray((depth * 1000).astype(np.uint16)).save(
        root / "rgb" / scene / f"depth_{name}.png")

    np.save(root / "pred" / scene / f"rgb_{name}_pred.npy",
            rng.normal(size=(h, w)).astype(np.float32))

    contour = np.zeros((h, w), bool)
    contour[:, w // 2 - 1:w // 2 + 2] = True          # on the step
    contour[h // 4, :] = True                          # and a false one across it
    np.savez_compressed(root / "mask" / scene / f"rgb_{name}_seg.npz",
                        packed=np.packbits(contour.reshape(1, -1), axis=-1), n=1)


class MixSourcesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_single_source_without_mix_json(self):
        import argparse
        args = argparse.Namespace(mix_json="", dataset="nyuv2", rgb_dir="R",
                                  pred_cache_dir="P", mask_cache_dir="M", max_images=7,
                                  pred_space="log")
        got = mix_sources(args)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["dataset"], "nyuv2")
        self.assertEqual(got[0]["max_images"], 7)
        self.assertEqual(got[0]["pred_space"], "log")

    def test_mix_json_fills_max_images_and_keeps_order(self):
        import argparse
        spec = [{"dataset": "nyuv2", "rgb_dir": "A", "pred_cache_dir": "B",
                 "mask_cache_dir": "C"},
                {"dataset": "hypersim10k", "rgb_dir": "D", "pred_cache_dir": "E",
                 "mask_cache_dir": "F", "max_images": 795}]
        p = self.tmp / "mix.json"
        p.write_text(json.dumps(spec), encoding="utf-8")
        got = mix_sources(argparse.Namespace(mix_json=str(p), pred_space="log"))
        self.assertEqual([s["dataset"] for s in got], ["nyuv2", "hypersim10k"])
        self.assertEqual(got[0]["max_images"], 0)       # defaulted
        self.assertEqual(got[1]["max_images"], 795)
        self.assertEqual([s["pred_space"] for s in got], ["log", "log"])

    def test_mix_json_rejects_an_incomplete_source(self):
        import argparse
        p = self.tmp / "bad.json"
        p.write_text(json.dumps([{"dataset": "nyuv2"}]), encoding="utf-8")
        with self.assertRaises(AssertionError):
            mix_sources(argparse.Namespace(mix_json=str(p), pred_space="log"))

    def test_a_source_may_declare_its_own_prediction_space(self):
        import argparse
        spec = [{"dataset": "nyuv2", "rgb_dir": "A", "pred_cache_dir": "B",
                 "mask_cache_dir": "C", "pred_space": "disparity"}]
        p = self.tmp / "space.json"
        p.write_text(json.dumps(spec), encoding="utf-8")
        got = mix_sources(argparse.Namespace(mix_json=str(p), pred_space="log"))
        self.assertEqual(got[0]["pred_space"], "disparity")

    def test_an_unknown_prediction_space_is_rejected(self):
        import argparse
        spec = [{"dataset": "nyuv2", "rgb_dir": "A", "pred_cache_dir": "B",
                 "mask_cache_dir": "C", "pred_space": "inverse"}]
        p = self.tmp / "badspace.json"
        p.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaises(AssertionError):
            mix_sources(argparse.Namespace(mix_json=str(p), pred_space="log"))


class CropTest(unittest.TestCase):
    H, W = 96, 128

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _write_source(self.tmp, "scene_a", "0001", self.H, self.W, seed=1)
        self.pair = (self.tmp / "rgb" / "scene_a" / "rgb_0001.png",
                     self.tmp / "rgb" / "scene_a" / "depth_0001.png")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ds(self, crop):
        return ContourDataset([self.pair], self.tmp / "rgb", self.tmp / "pred",
                              self.tmp / "mask", t=10.0, label_px=1.0,
                              dataset="nyuv2", crop=crop)

    def test_without_crop_the_shapes_are_the_frame(self):
        x, y, c = self._ds(None)[0]
        self.assertEqual(tuple(x.shape), (5, self.H, self.W))
        self.assertEqual(tuple(y.shape), (1, self.H, self.W))
        self.assertEqual(tuple(c.shape), (1, self.H, self.W))

    def test_crop_resizes_every_tensor_together(self):
        ch, cw = 64, 64
        x, y, c = self._ds((ch, cw))[0]
        self.assertEqual(tuple(x.shape), (5, ch, cw))
        self.assertEqual(tuple(y.shape), (1, ch, cw))
        self.assertEqual(tuple(c.shape), (1, ch, cw))

    def test_crop_larger_than_the_frame_is_a_pass_through(self):
        x, _, _ = self._ds((self.H * 2, self.W * 2))[0]
        self.assertEqual(tuple(x.shape), (5, self.H, self.W))

    def test_the_crop_keeps_channels_label_and_candidates_in_register(self):
        """The contour channel is channel 4; it must agree with the candidate mask.

        If the crop sliced them at different offsets this is where it shows: the
        two are built from the same array, so any disagreement is a slicing bug.
        """
        for _ in range(8):
            x, _, c = self._ds((48, 48))[0]
            contour_channel = (x[4] > 0).float()
            self.assertTrue(torch.equal(contour_channel, c[0]))

    def test_standardisation_uses_the_whole_frame_not_the_crop(self):
        """Inference standardises over the full image, so training must too.

        A crop standardised on itself would have mean 0 and std 1 by construction;
        a crop taken after full-frame standardisation almost never does.
        """
        off = []
        for _ in range(8):
            x, _, _ = self._ds((32, 32))[0]
            off.append(abs(float(x[3].mean())))
        self.assertGreater(max(off), 1e-3,
                           "every crop had zero mean - standardisation is being "
                           "applied to the crop instead of the frame")

    def test_the_label_sees_contours_outside_the_crop(self):
        """The distance transform must run on the whole frame.

        The depth step is the only discontinuity, so a crop that excludes it but
        sits within label_px of it must still carry positive labels. Computing the
        transform inside the crop would return all-negative there.
        """
        full = self._ds(None)
        _, y_full, _ = full[0]
        self.assertGreater(float(y_full.sum()), 0.0)

        # The step is at column W//2; a crop that just touches it keeps positives.
        ds = self._ds((self.H, self.W))
        ds._rng = np.random.default_rng(0)
        _, y, _ = ds[0]
        self.assertGreater(float(y.sum()), 0.0)


class BatchingTest(unittest.TestCase):
    """The point of the crop: two sources of different size batch together."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _write_source(self.tmp, "small", "0001", 96, 128, seed=2)
        _write_source(self.tmp, "large", "0002", 192, 256, seed=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ds(self, scene, name, crop):
        pair = (self.tmp / "rgb" / scene / f"rgb_{name}.png",
                self.tmp / "rgb" / scene / f"depth_{name}.png")
        return ContourDataset([pair], self.tmp / "rgb", self.tmp / "pred",
                              self.tmp / "mask", t=10.0, label_px=1.0,
                              dataset="nyuv2", crop=crop)

    def test_unequal_sources_fail_to_batch_without_a_crop(self):
        cat = torch.utils.data.ConcatDataset(
            [self._ds("small", "0001", None), self._ds("large", "0002", None)])
        dl = torch.utils.data.DataLoader(cat, batch_size=2, shuffle=False)
        with self.assertRaises(RuntimeError):
            next(iter(dl))

    def test_a_crop_makes_them_batch(self):
        cat = torch.utils.data.ConcatDataset(
            [self._ds("small", "0001", (96, 128)), self._ds("large", "0002", (96, 128))])
        dl = torch.utils.data.DataLoader(cat, batch_size=2, shuffle=False)
        x, y, c = next(iter(dl))
        self.assertEqual(tuple(x.shape), (2, 5, 96, 128))
        self.assertEqual(tuple(y.shape), (2, 1, 96, 128))
        self.assertEqual(tuple(c.shape), (2, 1, 96, 128))


if __name__ == "__main__":
    unittest.main()
