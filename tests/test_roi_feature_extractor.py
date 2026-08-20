import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from utils.roi_feature_extractor import (
    RoiFeatureCache,
    RoiFeatureExtractor,
    crop_bbox_square_padded,
)


class RoiFeatureExtractorTest(unittest.TestCase):
    def test_crop_bbox_square_padded(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[20:60, 30:70] = 255  # 40x40 square in center
        bbox = [30, 20, 70, 60]
        crop = crop_bbox_square_padded(img, bbox, target_size=224)
        self.assertEqual(crop.shape, (224, 224, 3))
        self.assertEqual(crop.dtype, np.uint8)

    def test_crop_bbox_normalized_coords(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        bbox = [0.15, 0.20, 0.35, 0.60]
        crop = crop_bbox_square_padded(img, bbox, target_size=224)
        self.assertEqual(crop.shape, (224, 224, 3))

    def test_crop_bbox_out_of_bounds(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bbox = [-10, 80, 50, 120]  # partially outside
        crop = crop_bbox_square_padded(img, bbox, target_size=224)
        self.assertEqual(crop.shape, (224, 224, 3))

    def test_extractor_single_and_batch(self):
        device = torch.device("cpu")
        extractor = RoiFeatureExtractor(device=device, weights=None)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox1 = [10, 10, 40, 50]
        bbox2 = [50, 50, 90, 90]

        feat1 = extractor.extract_single(img, bbox1)
        self.assertEqual(feat1.shape, (512,))

        feats = extractor.extract_batch(img, [bbox1, bbox2], batch_size=2)
        self.assertEqual(feats.shape, (2, 512))
        np.testing.assert_allclose(feat1, feats[0], atol=1e-5)

    def test_roi_feature_cache_save_and_load(self):
        keys = np.array(["scene1/img1.png", "scene1/img2.png"])
        offsets = np.array([0, 2, 3])
        roi_feats = np.random.randn(3, 512).astype(np.float32)
        metadata = {"backbone": "resnet18", "feature_dim": 512}

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "cache.npz"
            import json

            np.savez_compressed(
                npz_path,
                image_keys=keys,
                offsets=offsets,
                roi_features=roi_feats,
                metadata_json=np.asarray(json.dumps(metadata)),
            )

            cache = RoiFeatureCache.load(npz_path)
            f1 = cache.get_features("scene1/img1.png")
            self.assertEqual(f1.shape, (2, 512))
            np.testing.assert_array_equal(f1, roi_feats[0:2])

            f2 = cache.get_features("D:/lotus/data/scene1/img2.png")
            self.assertEqual(f2.shape, (1, 512))
            np.testing.assert_array_equal(f2, roi_feats[2:3])


if __name__ == "__main__":
    unittest.main()
