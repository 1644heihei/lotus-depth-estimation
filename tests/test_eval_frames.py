"""Tests for the per-dataset helpers that the single scoring path depends on.

scene_of is the one with teeth. ScanNet stores `scene0406_02/color/000300.jpg`, so taking
the parent directory as the scene names every frame `color` and collapses 271 scenes into
one. A scene-level split built that way is not a split at all: frames from the same room
land on both sides, validation precision reads higher than the model deserves, and
nothing fails. It went unnoticed in this project until a ScanNet training run was set up.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.eval_frames import (DATASETS, DEPTH_SCALE, MAX_DEPTH, depth_scale,
                               scene_of, valid_mask)


class SceneOfTest(unittest.TestCase):
    def test_scannet_reads_through_the_color_directory(self):
        p = Path("D:/data/scannet/val/scene0406_02/color/000300.jpg")
        self.assertEqual(scene_of("scannet", p), "scene0406_02")

    def test_nyuv2_and_hypersim_use_the_parent(self):
        self.assertEqual(
            scene_of("nyuv2", Path("/d/nyu/test/bathroom_0003/rgb_0046.png")),
            "bathroom_0003")
        self.assertEqual(
            scene_of("hypersim", Path("/d/hs/eval/ai_001_001/rgb_cam_00_fr0000.png")),
            "ai_001_001")

    def test_scannet_frames_from_one_room_share_a_scene(self):
        a = Path("/d/val/scene0406_02/color/000300.jpg")
        b = Path("/d/val/scene0406_02/color/000900.jpg")
        self.assertEqual(scene_of("scannet", a), scene_of("scannet", b))

    def test_scannet_frames_from_two_rooms_do_not(self):
        a = Path("/d/val/scene0406_02/color/000300.jpg")
        b = Path("/d/val/scene0088_02/color/000300.jpg")
        self.assertNotEqual(scene_of("scannet", a), scene_of("scannet", b))

    def test_the_parent_directory_would_have_collapsed_them(self):
        """The bug, stated as a test: this is what the old code returned."""
        a = Path("/d/val/scene0406_02/color/000300.jpg")
        b = Path("/d/val/scene0088_02/color/000300.jpg")
        self.assertEqual(a.parent.name, b.parent.name)
        self.assertEqual(a.parent.name, "color")

    def test_every_dataset_resolves_to_something(self):
        p = Path("/root/scene/sub/frame.png")
        for ds in DATASETS:
            self.assertTrue(scene_of(ds, p))


class DepthScaleTest(unittest.TestCase):
    def test_every_dataset_has_a_scale_and_a_range(self):
        for ds in DATASETS:
            self.assertIn(ds, DEPTH_SCALE)
            self.assertIn(ds, MAX_DEPTH)
            self.assertGreater(depth_scale(ds), 0)


class ValidMaskTest(unittest.TestCase):
    def test_zero_and_out_of_range_depths_are_invalid(self):
        gt = np.array([[0.0, 1.0, 5.0, 999.0]], dtype=np.float64)
        m = valid_mask("scannet", gt)
        self.assertFalse(m[0, 0], "zero depth accepted")
        self.assertTrue(m[0, 1])
        self.assertTrue(m[0, 2])
        self.assertFalse(m[0, 3], "depth beyond the dataset's range accepted")

    def test_non_finite_depths_are_invalid(self):
        gt = np.array([[np.nan, np.inf, 2.0]], dtype=np.float64)
        m = valid_mask("scannet", gt)
        self.assertFalse(m[0, 0])
        self.assertFalse(m[0, 1])
        self.assertTrue(m[0, 2])

    def test_nyuv2_applies_the_eigen_crop(self):
        """NYUv2 scores inside the Eigen crop by convention; the border is excluded."""
        gt = np.full((480, 640), 2.0)
        m = valid_mask("nyuv2", gt)
        self.assertFalse(m[0, 0], "top-left corner inside the Eigen crop")
        self.assertTrue(m[240, 320], "image centre excluded")
        self.assertLess(m.mean(), 1.0)


if __name__ == "__main__":
    unittest.main()
