import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from utils.object_depth_record import ObjectDepthRecord
from utils.object_depth_regressor import (
    BaselinePredictor,
    LinearDepthPredictor,
    ObjectDepthMLP,
    ObjectDepthRegressorBundle,
    ObjectDepthTabularDataset,
    RecordWithImage,
    compute_depth_metrics,
    predict_mlp_depth_m,
    save_model_bundle,
)


class ObjectDepthRegressorTest(unittest.TestCase):
    @staticmethod
    def _dummy_record(depth: float = 2.0, class_id: int = 1) -> ObjectDepthRecord:
        return ObjectDepthRecord(
            class_id=class_id,
            class_name="chair",
            bbox_w=0.2,
            bbox_h=0.3,
            bbox_area=0.06,
            bbox_cy_norm=0.5,
            depth_gt_m=depth,
            disparity_gt=1.0 / depth,
            score=0.9,
            x1=10,
            y1=20,
            x2=50,
            y2=80,
            mask_fill_ratio=0.8,
            mask_axis_ratio=1.5,
            mask_angle_sin2=0.5,
            mask_angle_cos2=0.5,
            mask_orientation_confidence=0.7,
            mask_valid=1.0,
        )

    def test_mlp_without_roi_features(self):
        mlp = ObjectDepthMLP(num_classes=91, embed_dim=16, feature_dim=13, hidden=(64, 32), roi_feature_dim=0)
        cid = torch.tensor([1, 2], dtype=torch.long)
        feat = torch.randn(2, 13)
        out = mlp(cid, feat)
        self.assertEqual(out.shape, (2,))

    def test_mlp_with_roi_features(self):
        mlp = ObjectDepthMLP(
            num_classes=91,
            embed_dim=16,
            feature_dim=13,
            hidden=(64, 32),
            roi_feature_dim=512,
            roi_projection_dim=64,
        )
        cid = torch.tensor([1, 2], dtype=torch.long)
        feat = torch.randn(2, 13)
        roi_feat = torch.randn(2, 512)

        out_with_roi = mlp(cid, feat, roi_features=roi_feat)
        self.assertEqual(out_with_roi.shape, (2,))

        # Test fallback when roi_features is None
        out_without_roi = mlp(cid, feat, roi_features=None)
        self.assertEqual(out_without_roi.shape, (2,))

    def test_dataset_with_roi_features(self):
        rec1 = self._dummy_record(depth=2.0)
        rec2 = self._dummy_record(depth=3.0)
        roi1 = np.random.randn(512).astype(np.float32)
        roi2 = np.random.randn(512).astype(np.float32)
        items = [
            RecordWithImage(record=rec1, rgb_path="img1.png", roi_feature=roi1),
            RecordWithImage(record=rec2, rgb_path="img2.png", roi_feature=roi2),
        ]
        ds = ObjectDepthTabularDataset(items, feature_version=3)
        self.assertEqual(len(ds), 2)
        sample = ds[0]
        self.assertIn("roi_features", sample)
        self.assertEqual(sample["roi_features"].shape, torch.Size([512]))

    def test_bundle_save_and_load_with_roi(self):
        mlp = ObjectDepthMLP(
            num_classes=91,
            embed_dim=16,
            feature_dim=13,
            hidden=(64, 32),
            roi_feature_dim=512,
            roi_projection_dim=64,
        )
        baseline = BaselinePredictor(global_median=3.0, per_class_median={1: 2.5})
        linear = LinearDepthPredictor(weights=np.zeros(91 + 13 + 1, dtype=np.float32), feature_dim=13)
        config = {
            "feature_version": 3,
            "feature_dim": 13,
            "embed_dim": 16,
            "hidden": [64, 32],
            "model_type": "mlp",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_bundle(tmpdir, mlp, baseline, linear, config)
            bundle = ObjectDepthRegressorBundle.load(tmpdir, device=torch.device("cpu"))
            self.assertEqual(bundle.config.get("roi_feature_dim"), 512)
            self.assertEqual(bundle.config.get("roi_projection_dim"), 64)

            rec = self._dummy_record()
            pred = bundle.predict_record(rec, roi_features=np.zeros(512, dtype=np.float32))
            self.assertIsInstance(pred, float)
            self.assertGreater(pred, 0.0)


if __name__ == "__main__":
    unittest.main()
