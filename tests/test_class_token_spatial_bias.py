"""Tests for the CLIP class-token spatial bias (option C of the text-conditioning plan).

Three properties decide whether the bias is safe to train with, and each has a way of
going wrong silently:

  shape        the bias is added to the attention logits, so a wrong Q or key width
               broadcasts instead of erroring and quietly biases the wrong thing
  zero columns tokens no detection claims - BOS, EOS, separators, padding - must stay
               exactly zero, or the bias reshapes attention to the whole prompt and the
               experiment stops measuring class semantics
  max          instances of one class collide on one token; a sum would double the bias
               where two objects overlap and a mean would pull the token to a midpoint
               between two chairs, which is a place neither chair is
"""

import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.object_spatial_attention import (
    build_class_token_spatial_bias,
    class_token_cross_attention_kwargs,
)

T = 77  # CLIP sequence width
INSIDE, OUTSIDE = 10.0, -2.0


def inputs(boxes, indices, valid=None, batch=1):
    k = len(boxes)
    bbox = torch.tensor([boxes], dtype=torch.float32).expand(batch, k, 4).contiguous()
    idx = torch.tensor([indices], dtype=torch.long).expand(batch, k).contiguous()
    m = torch.ones(batch, k, dtype=torch.bool) if valid is None else torch.tensor(
        [valid], dtype=torch.bool
    ).expand(batch, k).contiguous()
    return bbox, idx, m


class TestShape(unittest.TestCase):
    def test_shape_is_batch_one_query_tokens(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3], batch=2)
        bias = build_class_token_spatial_bias(bbox, idx, m, 8, 8, T)
        self.assertEqual(tuple(bias.shape), (2, 1, 64, T))

    def test_non_square_grid(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3])
        bias = build_class_token_spatial_bias(bbox, idx, m, 6, 8, T)
        self.assertEqual(tuple(bias.shape), (1, 1, 48, T))

    def test_every_latent_scale_the_unet_uses(self):
        # 512 training gives 64/32/16/8; 768 inference gives 96/48/24/12
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3])
        for s in (64, 48, 32, 24, 16, 12, 8):
            bias = build_class_token_spatial_bias(bbox, idx, m, s, s, T)
            self.assertEqual(tuple(bias.shape), (1, 1, s * s, T))

    def test_none_when_nothing_valid(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3], valid=[False])
        self.assertIsNone(build_class_token_spatial_bias(bbox, idx, m, 8, 8, T))


class TestZeroColumns(unittest.TestCase):
    def test_unclaimed_tokens_are_exactly_zero(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3])
        bias = build_class_token_spatial_bias(bbox, idx, m, 8, 8, T)[0, 0]
        claimed = torch.zeros(T, dtype=torch.bool)
        claimed[3] = True
        self.assertTrue(torch.all(bias[:, ~claimed] == 0.0))
        self.assertFalse(torch.all(bias[:, 3] == 0.0))

    def test_padding_rows_claim_nothing(self):
        # A padded row still carries index 0; it must not bias the BOS column.
        bbox, idx, m = inputs(
            [[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]], [5, 0], valid=[True, False]
        )
        bias = build_class_token_spatial_bias(bbox, idx, m, 8, 8, T)[0, 0]
        self.assertTrue(torch.all(bias[:, 0] == 0.0))
        self.assertFalse(torch.all(bias[:, 5] == 0.0))

    def test_multi_token_name_claims_every_one_of_its_tokens(self):
        box = [0.5, 0.5, 0.2, 0.2]
        bbox, idx, m = inputs([box, box], [4, 5])
        bias = build_class_token_spatial_bias(bbox, idx, m, 8, 8, T)[0, 0]
        torch.testing.assert_close(bias[:, 4], bias[:, 5])
        self.assertFalse(torch.all(bias[:, 4] == 0.0))


class TestValues(unittest.TestCase):
    def test_peak_sits_on_the_box_and_falls_off_to_the_corner(self):
        # Asserted against the grid rather than against INSIDE itself: cell centres land
        # at (i+0.5)/n, so no cell ever samples the box centre exactly and the peak comes
        # in a little under inside_bias (8.89 of 10.0 at 16x16 with a 0.2-wide box).
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3])
        grid = build_class_token_spatial_bias(bbox, idx, m, 16, 16, T)[0, 0, :, 3].reshape(16, 16)
        peak = divmod(int(grid.argmax()), 16)
        self.assertIn(peak, {(7, 7), (7, 8), (8, 7), (8, 8)})
        self.assertGreater(grid[peak].item(), 0.5 * (INSIDE + OUTSIDE))
        self.assertLess(grid[0, 0].item(), OUTSIDE * 0.9)

    def test_bias_stays_within_its_two_endpoints(self):
        bbox, idx, m = inputs([[0.3, 0.7, 0.4, 0.2]], [9])
        bias = build_class_token_spatial_bias(bbox, idx, m, 16, 16, T)
        self.assertLessEqual(bias.max().item(), INSIDE + 1e-4)
        self.assertGreaterEqual(bias[..., 9].min().item(), OUTSIDE - 1e-4)

    def test_custom_endpoints_are_honoured(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3])
        bias = build_class_token_spatial_bias(
            bbox, idx, m, 16, 16, T, inside_bias=4.0, outside_bias=-1.0
        )
        self.assertLessEqual(bias.max().item(), 4.0 + 1e-4)


class TestInstanceCollision(unittest.TestCase):
    def test_two_instances_take_the_max_not_the_sum(self):
        # Two chairs in opposite corners share one token. Each corner should read close to
        # a single chair's inside value - a sum would reach ~2x, a mean would leave both
        # corners weak and put the strength between them where no chair is.
        left = [0.2, 0.2, 0.2, 0.2]
        right = [0.8, 0.8, 0.2, 0.2]
        one, i1, m1 = inputs([left], [3])
        both, i2, m2 = inputs([left, right], [3, 3])

        single = build_class_token_spatial_bias(one, i1, m1, 16, 16, T)[0, 0, :, 3].reshape(16, 16)
        pair = build_class_token_spatial_bias(both, i2, m2, 16, 16, T)[0, 0, :, 3].reshape(16, 16)

        torch.testing.assert_close(pair[3, 3], single[3, 3], atol=1e-4, rtol=0)
        self.assertGreater(pair[12, 12].item(), single[12, 12].item())
        self.assertLessEqual(pair.max().item(), INSIDE + 1e-4)

    def test_result_is_independent_of_row_order(self):
        a = build_class_token_spatial_bias(
            *inputs([[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]], [3, 3]), 16, 16, T
        )
        b = build_class_token_spatial_bias(
            *inputs([[0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]], [3, 3]), 16, 16, T
        )
        torch.testing.assert_close(a, b)

    def test_different_classes_do_not_interfere(self):
        bbox, idx, m = inputs([[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]], [3, 7])
        bias = build_class_token_spatial_bias(bbox, idx, m, 16, 16, T)[0, 0]
        near = bias[:, 3].reshape(16, 16)
        far = bias[:, 7].reshape(16, 16)
        self.assertGreater(near[3, 3].item(), near[12, 12].item())
        self.assertGreater(far[12, 12].item(), far[3, 3].item())


class TestKwargs(unittest.TestCase):
    def test_disabled_returns_empty(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3])
        self.assertEqual(class_token_cross_attention_kwargs(bbox, idx, m, T, enabled=False), {})

    def test_all_padding_returns_empty(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3], valid=[False])
        self.assertEqual(class_token_cross_attention_kwargs(bbox, idx, m, T), {})

    def test_enabled_passes_inputs_not_a_prebuilt_bias(self):
        bbox, idx, m = inputs([[0.5, 0.5, 0.2, 0.2]], [3])
        kw = class_token_cross_attention_kwargs(bbox, idx, m, T)
        self.assertEqual(
            set(kw),
            {"class_token_bbox", "class_token_index", "class_token_mask",
             "class_token_num_text_tokens"},
        )
        self.assertEqual(kw["class_token_num_text_tokens"], T)


if __name__ == "__main__":
    unittest.main()
