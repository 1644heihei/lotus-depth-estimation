"""Tests for the dataset path registry.

Two opposite failures to guard, and the registry is only useful if it avoids both.

  - Overriding too little. The scripts' historical defaults are not empty: they name
    NYUv2 and Marigold V2. A registry that fills "only what is unset" would leave
    `--dataset hypersim --model marigold_v2` reading NYUv2 frames at the wrong
    resolution and never say so. That is the exact mistake this exists to stop - a
    Hypersim sweep in this project was first launched that way.
  - Overriding too much. Every command recorded in docs/ passes its paths explicitly.
    If the registry replaced those, the recorded numbers would stop reproducing.

So: a flag always beats the registry, and the registry always beats a default.
"""

import argparse
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.dataset_paths import (ALIGN_SPACE, DEFAULT_SPLIT, MODELS, PRED_CACHES,
                                 PRED_SPACE, RGB_DIRS, SAM_DIRS, pred_dir, resolve,
                                 splits_of, verify)
from utils.eval_frames import DATASETS, add_dataset_args


def _parser(with_res=False):
    """A parser shaped like the real scripts', defaults included."""
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_dir", default="NYU_DEFAULT")
    p.add_argument("--pred_cache_dir", default="MARIGOLD_DEFAULT")
    p.add_argument("--mask_cache_dir", default="SAM_DEFAULT")
    p.add_argument("--pred_space", default="log")
    p.add_argument("--align_space", default="log")
    if with_res:
        p.add_argument("--processing_res", type=int, default=640)
    add_dataset_args(p)
    return p


class RegistryShapeTest(unittest.TestCase):
    def test_every_dataset_has_a_default_split_with_paths(self):
        for ds in DATASETS:
            self.assertIn(ds, DEFAULT_SPLIT, f"{ds} has no default split")
            split = DEFAULT_SPLIT[ds]
            self.assertIn((ds, split), RGB_DIRS, f"{ds}/{split} has no RGB dir")
            self.assertIn((ds, split), SAM_DIRS, f"{ds}/{split} has no SAM dir")

    def test_every_rgb_dir_has_a_matching_sam_dir(self):
        self.assertEqual(set(RGB_DIRS), set(SAM_DIRS))

    def test_every_prediction_cache_names_a_known_dataset_split_and_model(self):
        for ds, split, model in PRED_CACHES:
            self.assertIn((ds, split), RGB_DIRS, f"{ds}/{split} has no frames")
            self.assertIn(model, MODELS)
            self.assertIn(model, PRED_SPACE)
            self.assertIn(model, ALIGN_SPACE)

    def test_pred_dir_joins_the_resolution(self):
        self.assertTrue(pred_dir("hypersim", "eval", "marigold_v2").endswith("/res1024"))
        self.assertTrue(pred_dir("nyuv2", "test", "lotus_v1").endswith("/res768"))

    def test_splits_of_lists_what_exists(self):
        self.assertEqual(set(splits_of("nyuv2")), {"test", "train"})


class ResolveTest(unittest.TestCase):
    def test_the_registry_beats_a_default(self):
        """The bug this registry exists to stop."""
        p = _parser(with_res=True)
        a = resolve(p.parse_args(["--dataset", "hypersim", "--model", "marigold_v2"]), p)
        self.assertEqual(a.rgb_dir, RGB_DIRS[("hypersim", "eval")])
        self.assertEqual(a.mask_cache_dir, SAM_DIRS[("hypersim", "eval")])
        self.assertEqual(a.processing_res, 1024)
        self.assertNotEqual(a.rgb_dir, "NYU_DEFAULT")

    def test_a_flag_beats_the_registry(self):
        """Every command recorded in docs/ passes its paths; they must survive."""
        p = _parser(with_res=True)
        a = resolve(p.parse_args(["--dataset", "hypersim", "--model", "marigold_v2",
                                  "--rgb_dir", "MINE", "--processing_res", "512"]), p)
        self.assertEqual(a.rgb_dir, "MINE")
        self.assertEqual(a.processing_res, 512)

    def test_no_model_leaves_the_prediction_cache_alone(self):
        p = _parser(with_res=True)
        a = resolve(p.parse_args([]), p)
        self.assertEqual(a.pred_cache_dir, "MARIGOLD_DEFAULT")

    def test_the_model_decides_the_prediction_space(self):
        p = _parser(with_res=True)
        a = resolve(p.parse_args(["--model", "lotus_v1"]), p)
        self.assertEqual(a.pred_space, "disparity")
        self.assertEqual(a.align_space, "disparity")
        self.assertEqual(a.processing_res, 768)

    def test_a_script_without_processing_res_gets_the_joined_path(self):
        p = _parser(with_res=False)
        a = resolve(p.parse_args(["--dataset", "scannet", "--model", "marigold_v2"]), p)
        self.assertTrue(a.pred_cache_dir.endswith("/res640"), a.pred_cache_dir)

    def test_an_unavailable_combination_fails_loudly(self):
        p = _parser(with_res=True)
        with self.assertRaises(AssertionError):
            resolve(p.parse_args(["--dataset", "hypersim", "--model", "lotus_v1"]), p)

    def test_split_selects_the_training_frames(self):
        p = _parser(with_res=True)
        a = resolve(p.parse_args(["--split", "train", "--model", "marigold_v2"]), p)
        self.assertEqual(a.rgb_dir, RGB_DIRS[("nyuv2", "train")])
        self.assertEqual(a.mask_cache_dir, SAM_DIRS[("nyuv2", "train")])

    def test_dataset_specific_extras_are_filled(self):
        p = _parser(with_res=True)
        a = resolve(p.parse_args(["--dataset", "scannet", "--model", "marigold_v2"]), p)
        self.assertTrue(a.scannet_root.endswith("scannet"), a.scannet_root)

    def test_resolve_without_a_parser_fills_only_unset_values(self):
        from types import SimpleNamespace
        a = SimpleNamespace(dataset="scannet", rgb_dir=None, mask_cache_dir="KEEP",
                            pred_cache_dir=None, pred_space=None, align_space=None)
        resolve(a, model="marigold_v2")
        self.assertEqual(a.rgb_dir, RGB_DIRS[("scannet", "val")])
        self.assertEqual(a.mask_cache_dir, "KEEP")


class OnDiskTest(unittest.TestCase):
    def test_the_registry_matches_the_disk(self):
        """A registry that is wrong is worse than retyping, because it is trusted.

        Skipped rather than failed off this machine: the paths are absolute and local.
        """
        missing = verify()
        if len(missing) == len(RGB_DIRS) + len(SAM_DIRS) + len(PRED_CACHES) + 4:
            self.skipTest("none of the data is present; not this machine")
        self.assertEqual(missing, [], "registered paths missing from disk")


if __name__ == "__main__":
    unittest.main()
