import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.build_object_depth_records import validate_dataset_depth_unit
from utils.object_depth_record import ObjectDepthRecord, load_gt_depth_meters
from utils.object_depth_regressor import (
    BaselinePredictor,
    ObjectDepthMLP,
    ObjectDepthRegressorBundle,
    RecordWithImage,
    extract_record_features,
    filter_training_items,
    scene_level_split,
)
from utils.object_detection_cache import ObjectDetection, mask_shape_features
from utils.object_pre_depth_regressor import (
    build_pre_depth_from_regressor,
    fit_relative_disparity_calibration,
)
from utils.object_pre_depth import load_pre_depth_artifacts, save_pre_depth_artifacts


class _DummyRegressor:
    def predict_depth_m(self, class_id, bbox_w, bbox_h, bbox_cy_norm, score=0.5, **kwargs):
        return {1: 2.0, 2: 4.0}.get(class_id, 3.0)


class ObjectDepthRegressorPipelineTest(unittest.TestCase):
    @staticmethod
    def _record(depth_m=2.0, score=0.75):
        return ObjectDepthRecord(
            class_id=1,
            class_name="chair",
            bbox_w=100,
            bbox_h=200,
            bbox_area=20000,
            bbox_cy_norm=0.5,
            score=score,
            depth_gt_m=depth_m,
            disparity_gt=1.0 / depth_m,
            x1=0,
            y1=0,
            x2=100,
            y2=200,
            dataset="test",
            depth_unit="m",
        )

    def test_explicit_depth_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.png"
            Image.fromarray(np.array([[1000, 2500]], dtype=np.uint16)).save(path)
            mm = load_gt_depth_meters(path, depth_unit="mm")
            cm = load_gt_depth_meters(path, depth_unit="cm")
            np.testing.assert_allclose(mm, [[1.0, 2.5]])
            np.testing.assert_allclose(cm, [[10.0, 25.0]])

    def test_known_dataset_rejects_wrong_depth_unit(self):
        validate_dataset_depth_unit("hypersim_train", "mm")
        with self.assertRaises(ValueError):
            validate_dataset_depth_unit("hypersim_train", "cm")

    def test_compact_artifact_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb_path = Path(tmp) / "rgb_0001.png"
            Image.new("RGB", (8, 6)).save(rgb_path)
            pre = np.linspace(-1, 1, 48, dtype=np.float32).reshape(6, 8)
            valid = np.ones((6, 8), dtype=np.float32)
            save_pre_depth_artifacts(
                pre,
                valid,
                rgb_path,
                Path(tmp) / "artifacts",
                storage_size=(4, 4),
                compact=True,
            )
            saved_pre, saved_valid = load_pre_depth_artifacts(
                rgb_path, Path(tmp) / "artifacts"
            )
            self.assertEqual(saved_pre.shape, (4, 4))
            self.assertEqual(saved_pre.dtype, np.float16)
            self.assertEqual(saved_valid.dtype, np.uint8)

    def test_shift_only_calibration_for_one_object(self):
        scale, shift = fit_relative_disparity_calibration(
            np.array([0.25]), np.array([0.6])
        )
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(scale * 0.25 + shift, 0.6)

    def test_v2_feature_contains_detection_score(self):
        rec = self._record(score=0.83)
        features = extract_record_features(rec, feature_version=2)
        self.assertAlmostEqual(float(features[-1]), 0.83)

    def test_v3_features_include_mask_orientation(self):
        rec = self._record(score=0.83)
        rec.mask_fill_ratio = 0.6
        rec.mask_axis_ratio = 3.0
        rec.mask_angle_sin2 = 0.8
        rec.mask_angle_cos2 = 0.6
        rec.mask_orientation_confidence = 0.7
        rec.mask_valid = 1.0
        features = extract_record_features(rec, feature_version=3)
        self.assertEqual(features.shape, (13,))
        np.testing.assert_allclose(features[-4:], [0.8, 0.6, 0.7, 1.0])

    def test_bundle_prediction_uses_mask_features(self):
        model = ObjectDepthMLP(embed_dim=1, feature_dim=13, hidden=(), dropout=0.0)
        with torch.no_grad():
            model.class_embed.weight.zero_()
            model.mlp[0].weight.zero_()
            model.mlp[0].bias.zero_()
            model.mlp[0].weight[0, 1 + 9] = 1.0  # mask_angle_sin2
        bundle = ObjectDepthRegressorBundle(
            mlp=model,
            baseline=BaselinePredictor(1.0, {}),
            linear=None,
            config={"feature_version": 3, "model_type": "mlp"},
            device=torch.device("cpu"),
        )
        default_depth = bundle.predict_depth_m(1, 0.2, 0.3)
        oriented_depth = bundle.predict_depth_m(
            1, 0.2, 0.3, mask_angle_sin2=0.8, mask_valid=1.0
        )
        self.assertAlmostEqual(default_depth, 1.0)
        self.assertAlmostEqual(oriented_depth, float(np.exp(0.8)), places=5)

    def test_mask_shape_features_capture_diagonal_axis(self):
        mask = np.zeros((32, 32), dtype=bool)
        for i in range(4, 28):
            mask[max(0, i - 1) : i + 2, max(0, i - 1) : i + 2] = True
        features = mask_shape_features(mask, bbox_area=32 * 32)
        self.assertGreater(features["mask_axis_ratio"], 2.0)
        self.assertGreater(features["mask_angle_sin2"], 0.8)
        self.assertEqual(features["mask_valid"], 1.0)

    def test_scene_split_keeps_frames_together(self):
        items = [
            RecordWithImage(self._record(), f"train/ai_001_001/rgb_{i}.png")
            for i in range(3)
        ] + [
            RecordWithImage(self._record(), f"train/ai_002_001/rgb_{i}.png")
            for i in range(3)
        ]
        train, val = scene_level_split(items, val_ratio=0.5, seed=42)
        train_scenes = {Path(item.rgb_path).parent.name for item in train}
        val_scenes = {Path(item.rgb_path).parent.name for item in val}
        self.assertTrue(train_scenes.isdisjoint(val_scenes))

    def test_depth_filter_is_reproducible(self):
        items = [
            RecordWithImage(self._record(depth_m=0.2), "a.png"),
            RecordWithImage(self._record(depth_m=2.0), "b.png"),
            RecordWithImage(self._record(depth_m=9.0), "c.png"),
        ]
        kept = filter_training_items(items, min_depth_m=0.3, max_depth_m=8.0)
        self.assertEqual([item.rgb_path for item in kept], ["b.png"])

    def test_positive_affine_calibration(self):
        scale, shift = fit_relative_disparity_calibration(
            np.array([0.1, 0.2, 0.4]), np.array([0.25, 0.4, 0.7])
        )
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(
            scale * np.array([0.1, 0.2, 0.4]) + shift,
            np.array([0.25, 0.4, 0.7]),
            atol=1e-6,
        )

    def test_zero_detection_returns_global_with_zero_mask(self):
        global_disp = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        pre, mask = build_pre_depth_from_regressor(
            global_disp, [], _DummyRegressor(), blend_blur_ksize=1
        )
        np.testing.assert_allclose(pre, global_disp * 2 - 1)
        self.assertEqual(float(mask.sum()), 0.0)

    def test_overlap_is_independent_of_detection_order(self):
        global_disp = np.full((16, 16), 0.5, dtype=np.float32)
        low = ObjectDetection([2, 2, 12, 12], 1, "low", 0.6)
        high = ObjectDetection([4, 4, 14, 14], 2, "high", 0.9)
        pre_a, _ = build_pre_depth_from_regressor(
            global_disp, [low, high], _DummyRegressor(), blend_blur_ksize=1
        )
        pre_b, _ = build_pre_depth_from_regressor(
            global_disp, [high, low], _DummyRegressor(), blend_blur_ksize=1
        )
        np.testing.assert_allclose(pre_a, pre_b)


if __name__ == "__main__":
    unittest.main()
