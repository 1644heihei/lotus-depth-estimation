import unittest

import torch

from utils.object_attention_condition import OBJECT_CONTINUOUS_FEATURE_DIM
from utils.object_spatial_attention import (
    build_object_spatial_attention_bias,
    object_cross_attention_kwargs,
)


class ObjectSpatialAttentionTest(unittest.TestCase):
    def test_bias_shape_and_object_only_keys(self):
        features = torch.tensor(
            [
                [
                    [0.5, 0.5, 0.2, 0.2] + [0.0] * (OBJECT_CONTINUOUS_FEATURE_DIM - 4),
                    [0.0, 0.0, 0.0, 0.0] + [0.0] * (OBJECT_CONTINUOUS_FEATURE_DIM - 4),
                ]
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True, False]])
        text = torch.randn(1, 4, 8)
        bias = build_object_spatial_attention_bias(
            features, mask, latent_height=8, latent_width=8, num_text_tokens=4
        )
        self.assertIsNotNone(bias)
        self.assertEqual(bias.shape, (1, 1, 64, 6))
        self.assertTrue(torch.allclose(bias[..., :4], torch.zeros_like(bias[..., :4])))
        self.assertFalse(torch.allclose(bias[..., 4], torch.zeros_like(bias[..., 4])))
        self.assertTrue(torch.allclose(bias[..., 5], torch.zeros_like(bias[..., 5])))

    def test_bias_none_when_no_objects(self):
        features = torch.zeros(1, 2, OBJECT_CONTINUOUS_FEATURE_DIM)
        mask = torch.zeros(1, 2, dtype=torch.bool)
        bias = build_object_spatial_attention_bias(
            features, mask, latent_height=4, latent_width=4, num_text_tokens=3
        )
        self.assertIsNone(bias)

    def test_cross_attention_kwargs_helper(self):
        features = torch.tensor(
            [[[0.5, 0.5, 0.4, 0.4] + [0.0] * (OBJECT_CONTINUOUS_FEATURE_DIM - 4)]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[True]])
        text = torch.randn(1, 2, 16)
        kwargs = object_cross_attention_kwargs(
            features, mask, text, latent_height=4, latent_width=4, enabled=True
        )
        self.assertIn("object_spatial_features", kwargs)
        self.assertIn("object_spatial_num_text_tokens", kwargs)
        self.assertEqual(kwargs["object_spatial_num_text_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
