"""Tests for the shared class-name prompt construction.

The point of utils/object_prompt.py is that training and evaluation build the identical
string and the identical token positions, so these tests pin the format and the token
matching rather than any downstream behaviour.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.object_detection_cache import ObjectDetection
from utils.object_prompt import (
    build_class_prompt,
    class_names,
    class_token_bias_inputs,
    class_token_spans,
)


def det(name, bbox=(0, 0, 10, 10), score=0.9, class_id=0):
    return ObjectDetection(bbox=list(bbox), class_id=class_id, class_name=name, score=score)


class FakeTokenizer:
    """Minimal stand-in: one id per word, BOS/EOS around the prompt, pad to max length.

    Enough to exercise span matching including multi-word class names, without pulling
    CLIP's weights into a unit test.
    """

    model_max_length = 12
    BOS, EOS, PAD = 100, 101, 0

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        out = []
        for word in text.replace(",", " ").split():
            out.append(self.vocab.setdefault(word, 200 + len(self.vocab)))
        return out

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None, add_special_tokens=True):
        ids = self._ids(text)
        if add_special_tokens:
            ids = [self.BOS] + ids + [self.EOS]
        if padding == "max_length":
            ids = (ids + [self.PAD] * max_length)[:max_length]

        class _Out:
            pass

        out = _Out()
        out.input_ids = np.array([ids]) if return_tensors else ids
        if return_tensors:
            out.input_ids = _TensorLike(ids)
        return out


class _TensorLike:
    def __init__(self, ids):
        self._ids = [ids]

    def __getitem__(self, i):
        return _ListLike(self._ids[i])


class _ListLike(list):
    def tolist(self):
        return list(self)


class TestClassPrompt(unittest.TestCase):
    def test_names_are_deduplicated_and_sorted(self):
        dets = [det("sink"), det("chair"), det("chair"), det("bed")]
        self.assertEqual(class_names(dets), ["bed", "chair", "sink"])

    def test_score_threshold_filters(self):
        dets = [det("chair", score=0.9), det("bed", score=0.2)]
        self.assertEqual(build_class_prompt(dets, score_thr=0.5), "chair")

    def test_prompt_format_is_comma_space(self):
        self.assertEqual(build_class_prompt([det("chair"), det("sink")]), "chair, sink")

    def test_no_detections_gives_empty_prompt(self):
        # Not a special case: 39.5% of Hypersim frames land here and empty is exactly
        # what Lotus was fine-tuned on.
        self.assertEqual(build_class_prompt([]), "")

    def test_order_is_independent_of_detection_order(self):
        a = build_class_prompt([det("sink"), det("chair")])
        b = build_class_prompt([det("chair"), det("sink")])
        self.assertEqual(a, b)


class TestTokenSpans(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_spans_point_at_the_right_positions(self):
        prompt = "chair, sink"
        spans = class_token_spans(self.tok, prompt, ["chair", "sink"])
        ids = self.tok(prompt, padding="max_length", max_length=12,
                       truncation=True, return_tensors="pt").input_ids[0].tolist()
        for name, span in spans.items():
            expected = self.tok(name, add_special_tokens=False).input_ids
            self.assertEqual([ids[i] for i in span], expected)

    def test_multi_word_class_spans_several_tokens(self):
        spans = class_token_spans(self.tok, "dining table, sink", ["dining table", "sink"])
        self.assertEqual(len(spans["dining table"]), 2)
        self.assertEqual(len(spans["sink"]), 1)

    def test_spans_skip_the_bos_token(self):
        spans = class_token_spans(self.tok, "chair", ["chair"])
        self.assertNotIn(0, spans["chair"])


class TestBiasInputs(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_box_is_normalised_centre_and_size(self):
        d = det("chair", bbox=(20, 40, 60, 80))
        bbox, _, valid, _ = class_token_bias_inputs(
            [d], self.tok, image_height=100, image_width=200
        )
        cx, cy, bw, bh = bbox[0]
        self.assertAlmostEqual(cx, 40 / 200, places=5)
        self.assertAlmostEqual(cy, 60 / 100, places=5)
        self.assertAlmostEqual(bw, 40 / 200, places=5)
        self.assertAlmostEqual(bh, 40 / 100, places=5)
        self.assertTrue(valid[0])

    def test_multi_token_name_repeats_the_box(self):
        d = det("dining table", bbox=(0, 0, 50, 50))
        bbox, index, valid, _ = class_token_bias_inputs(
            [d], self.tok, image_height=100, image_width=100
        )
        self.assertEqual(int(valid.sum()), 2)
        np.testing.assert_allclose(bbox[0], bbox[1])
        self.assertNotEqual(index[0], index[1])

    def test_two_instances_of_one_class_share_a_token_index(self):
        # The consumer resolves the collision with a max, so the duplicate index is the
        # intended representation of "attend wherever either of them is".
        dets = [det("chair", bbox=(0, 0, 10, 10)), det("chair", bbox=(80, 80, 100, 100))]
        bbox, index, valid, _ = class_token_bias_inputs(
            dets, self.tok, image_height=100, image_width=100
        )
        self.assertEqual(int(valid.sum()), 2)
        self.assertEqual(index[0], index[1])
        self.assertFalse(np.allclose(bbox[0], bbox[1]))

    def test_no_detections_gives_all_invalid(self):
        bbox, index, valid, prompt = class_token_bias_inputs(
            [], self.tok, image_height=100, image_width=100
        )
        self.assertEqual(prompt, "")
        self.assertFalse(valid.any())
        self.assertEqual(bbox.shape, (16, 4))

    def test_cap_truncates_instead_of_failing(self):
        dets = [det(f"c{i}", bbox=(0, 0, 10, 10)) for i in range(20)]
        _, _, valid, _ = class_token_bias_inputs(
            dets, self.tok, image_height=100, image_width=100, max_tokens=4
        )
        self.assertEqual(int(valid.sum()), 4)

    def test_prompt_matches_build_class_prompt(self):
        # The whole reason this module exists: one string, one definition.
        dets = [det("sink"), det("chair")]
        _, _, _, prompt = class_token_bias_inputs(
            dets, self.tok, image_height=100, image_width=100
        )
        self.assertEqual(prompt, build_class_prompt(dets))


if __name__ == "__main__":
    unittest.main()
