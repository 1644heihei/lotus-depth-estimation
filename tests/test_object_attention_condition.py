import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from utils.object_attention_condition import (
    OBJECT_CONTINUOUS_FEATURE_DIM,
    ObjectAttentionCache,
    ObjectAttentionEncoder,
    append_object_attention_tokens,
    apply_object_condition_dropout,
    encode_object_attention_condition,
    object_condition_encoder_spatial_bias_enabled,
    object_rows_from_detections,
    padded_object_condition_from_detections,
    stack_object_rows,
    validate_object_attention_cache_metadata,
)
from utils.object_detection_cache import ObjectDetection


class ObjectAttentionConditionTest(unittest.TestCase):
    @staticmethod
    def _detection(score=0.8):
        return ObjectDetection(
            bbox=[10, 20, 50, 80],
            class_id=3,
            class_name="object",
            score=score,
            mask_fill_ratio=0.6,
            mask_axis_ratio=2.0,
            mask_angle_sin2=0.8,
            mask_angle_cos2=0.6,
            mask_orientation_confidence=0.5,
            mask_valid=1.0,
        )

    def test_cache_top_k_and_padding(self):
        features = np.arange(
            3 * OBJECT_CONTINUOUS_FEATURE_DIM, dtype=np.float32
        ).reshape(3, -1)
        keys, offsets, classes, all_features = stack_object_rows(
            [("C:/data/a.png", [2, 3, 4], features), ("C:/data/b.png", [], [])]
        )
        cache = ObjectAttentionCache(keys, offsets, classes, all_features, {})
        ids, padded, mask = cache.padded("c:\\data\\a.png", 2)
        np.testing.assert_array_equal(ids, [2, 3])
        np.testing.assert_array_equal(padded, features[:2])
        np.testing.assert_array_equal(mask, [True, True])
        _, _, empty_mask = cache.padded("C:/data/b.png", 2)
        self.assertFalse(empty_mask.any())

    def test_padded_for_eval_resolves_relative_path(self):
        features = np.arange(
            2 * OBJECT_CONTINUOUS_FEATURE_DIM, dtype=np.float32
        ).reshape(2, -1)
        keys, offsets, classes, all_features = stack_object_rows(
            [
                (
                    "C:/datasets/eval/nyuv2/test/bathroom/rgb_00001.png",
                    [7, 8],
                    features,
                )
            ]
        )
        cache = ObjectAttentionCache(keys, offsets, classes, all_features, {})
        ids, padded, mask = cache.padded_for_eval(
            "bathroom/rgb_00001.png",
            4,
            rgb_root="C:/datasets/eval/nyuv2/test",
        )
        np.testing.assert_array_equal(ids[:2], [7, 8])
        np.testing.assert_array_equal(padded[:2], features)
        np.testing.assert_array_equal(mask[:2], [True, True])
        self.assertFalse(mask[2:].any())

    def test_object_count_from_cache(self):
        features = np.arange(
            3 * OBJECT_CONTINUOUS_FEATURE_DIM, dtype=np.float32
        ).reshape(3, -1)
        keys, offsets, classes, all_features = stack_object_rows(
            [
                ("C:/data/a.png", [2, 3, 4], features),
                ("C:/data/b.png", [], []),
            ]
        )
        cache = ObjectAttentionCache(keys, offsets, classes, all_features, {})
        self.assertEqual(cache.object_count("C:/data/a.png"), 3)
        self.assertEqual(cache.object_count("C:/data/b.png"), 0)
        self.assertEqual(
            cache.object_count("b.png", rgb_root="C:/data"),
            0,
        )

    def test_encoder_shapes_and_mask(self):
        encoder = ObjectAttentionEncoder(
            class_embed_dim=8,
            hidden_dim=16,
            num_hidden_layers=2,
            cross_attention_dim=32,
        )
        ids = torch.tensor([[1, 2, 0]])
        features = torch.randn(1, 3, OBJECT_CONTINUOUS_FEATURE_DIM)
        mask = torch.tensor([[True, True, False]])
        tokens = encoder(ids, features, mask)
        self.assertEqual(tokens.shape, (1, 3, 32))
        self.assertTrue(torch.equal(tokens[:, 2], torch.zeros_like(tokens[:, 2])))

    def test_encoder_zero_init_output(self):
        encoder = ObjectAttentionEncoder(
            class_embed_dim=8,
            hidden_dim=16,
            num_hidden_layers=3,
            cross_attention_dim=32,
        )
        ids = torch.tensor([[1]])
        features = torch.randn(1, 1, OBJECT_CONTINUOUS_FEATURE_DIM)
        mask = torch.ones(1, 1, dtype=torch.bool)
        tokens = encoder(ids, features, mask)
        self.assertEqual(tokens.shape, (1, 1, 32))
        self.assertTrue(torch.allclose(tokens, torch.zeros_like(tokens), atol=1e-5))

    def test_shared_encode_and_append_path(self):
        encoder = ObjectAttentionEncoder(
            class_embed_dim=8,
            hidden_dim=16,
            num_hidden_layers=2,
            cross_attention_dim=32,
        )
        text = torch.randn(1, 4, 32)
        ids = torch.tensor([[1, 0]])
        features = torch.randn(1, 2, OBJECT_CONTINUOUS_FEATURE_DIM)
        mask = torch.tensor([[True, False]])
        states, combined_mask = encode_object_attention_condition(
            text, encoder, ids, features, mask
        )
        self.assertEqual(states.shape, (1, 6, 32))
        self.assertTrue(torch.equal(combined_mask[:, 4:], mask))

    def test_append_tokens_and_padding_mask(self):
        text = torch.randn(2, 4, 8)
        objects = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, False, False], [True, True, False]])
        states, combined_mask = append_object_attention_tokens(text, objects, mask)
        self.assertEqual(states.shape, (2, 7, 8))
        self.assertEqual(combined_mask.shape, (2, 7))
        self.assertTrue(combined_mask[:, :4].all())
        self.assertTrue(torch.equal(combined_mask[:, 4:], mask))

    def test_empty_objects_preserve_text_path(self):
        text = torch.randn(1, 4, 8)
        states, mask = append_object_attention_tokens(
            text, torch.zeros(1, 2, 8), torch.zeros(1, 2, dtype=torch.bool)
        )
        self.assertIs(states, text)
        self.assertIsNone(mask)

    def test_dropout_is_per_image(self):
        mask = torch.ones(4, 3, dtype=torch.bool)
        self.assertFalse(
            apply_object_condition_dropout(mask, 1.0, training=True).any()
        )
        self.assertTrue(
            apply_object_condition_dropout(mask, 1.0, training=False).all()
        )

    def test_encoder_save_load(self):
        encoder = ObjectAttentionEncoder(
            class_embed_dim=8,
            hidden_dim=16,
            num_hidden_layers=3,
            cross_attention_dim=32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            encoder.save_pretrained(tmp)
            loaded = ObjectAttentionEncoder.from_pretrained(tmp)
            self.assertEqual(loaded.config.cross_attention_dim, 32)
            self.assertEqual(
                set(encoder.state_dict().keys()), set(loaded.state_dict().keys())
            )

    def test_cache_and_online_condition_are_identical(self):
        class Regressor:
            def predict_depth_m(self, *args, **kwargs):
                return 2.5

        detections = [self._detection()]
        classes, features = object_rows_from_detections(
            detections, 100, 100, Regressor()
        )
        keys, offsets, all_classes, all_features = stack_object_rows(
            [("image.png", classes, features)]
        )
        cache = ObjectAttentionCache(
            keys, offsets, all_classes, all_features, {}
        )
        cached_ids, cached_features, cached_mask = cache.padded("image.png", 4)
        ids, online_features, mask = padded_object_condition_from_detections(
            detections, 100, 100, Regressor(), 4
        )
        np.testing.assert_array_equal(cached_ids, ids.numpy()[0])
        np.testing.assert_allclose(cached_features, online_features.numpy()[0])
        np.testing.assert_array_equal(cached_mask, mask.numpy()[0])

    def test_cache_rejects_invalid_offsets(self):
        with self.assertRaises(ValueError):
            ObjectAttentionCache(
                image_keys=np.asarray(["image.png"]),
                offsets=np.asarray([0, 2]),
                class_ids=np.asarray([1]),
                features=np.zeros(
                    (1, OBJECT_CONTINUOUS_FEATURE_DIM), dtype=np.float32
                ),
                metadata={},
            )

    def test_validate_cache_metadata_regressor_mismatch(self):
        with self.assertRaises(ValueError):
            validate_object_attention_cache_metadata(
                {"regressor_dir": "D:/lotus/regressor_a"},
                regressor_dir="D:/lotus/regressor_b",
            )

    def test_roi_regressor_requires_rgb_for_online_features(self):
        class RoiRegressor:
            config = {"roi_feature_dim": 32}

        with self.assertRaises(ValueError):
            padded_object_condition_from_detections(
                [self._detection()],
                100,
                100,
                RoiRegressor(),
                4,
            )

    def test_spatial_bias_flag_from_encoder_config(self):
        enabled = ObjectAttentionEncoder(
            class_embed_dim=8,
            hidden_dim=16,
            num_hidden_layers=3,
            cross_attention_dim=32,
            enable_object_spatial_bias=True,
        )
        disabled = ObjectAttentionEncoder(
            class_embed_dim=8,
            hidden_dim=16,
            num_hidden_layers=3,
            cross_attention_dim=32,
            enable_object_spatial_bias=False,
        )
        legacy = ObjectAttentionEncoder(
            class_embed_dim=8,
            hidden_dim=16,
            num_hidden_layers=2,
            cross_attention_dim=32,
            enable_object_spatial_bias=False,
        )
        self.assertTrue(
            object_condition_encoder_spatial_bias_enabled(enabled)
        )
        self.assertFalse(
            object_condition_encoder_spatial_bias_enabled(disabled)
        )
        self.assertFalse(object_condition_encoder_spatial_bias_enabled(legacy))
        self.assertFalse(object_condition_encoder_spatial_bias_enabled(None))


if __name__ == "__main__":
    unittest.main()
