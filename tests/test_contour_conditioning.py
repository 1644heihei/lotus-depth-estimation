"""Tests for the contour-conditioning path, aimed at the failure it already had.

The first attempt fed contours through three 80-minute runs without them ever reaching the
output: the expansion zero-initialised the new input slices, LoRA froze everything it did
not target, and the slices stayed at zero. Nothing in the loss or the logs showed it,
because logging an input proves the tensor was passed, not that it changed anything.

So these check the property that actually matters - the conditioning can alter the output -
plus the two mechanics that broke it, and the mask decoding that produced empty contours
before that.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.contour_condition import ContourCache, contour_of
from utils.expanded_conv_in import (
    FILENAME,
    extra_channel_energy,
    load_conv_in,
    save_conv_in,
    unfreeze_conv_in,
)
from utils.pre_depth_fusion import expand_unet_conv_in


class _Config:
    """expand_unet_conv_in keeps unet.config.in_channels in step with the layer."""

    def __init__(self, in_channels):
        self.in_channels = in_channels


class TinyUNet(torch.nn.Module):
    """Just enough surface for the conv_in helpers: a conv_in and a forward through it."""

    def __init__(self, in_ch=4, out_ch=8):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.config = _Config(in_ch)

    def forward(self, x):
        return self.conv_in(x)


class TestExpansionIsTrainable(unittest.TestCase):
    def test_expansion_starts_at_zero(self):
        u = TinyUNet()
        expand_unet_conv_in(u, 4, zero_init=True)
        self.assertEqual(u.conv_in.in_channels, 8)
        self.assertEqual(extra_channel_energy(u, 4), 0.0)

    def test_zero_slices_make_the_extra_channels_inert(self):
        # The bug, stated as a test: with the slices at zero, changing the conditioning
        # cannot change the output no matter what is fed.
        u = TinyUNet()
        expand_unet_conv_in(u, 4, zero_init=True)
        img = torch.randn(1, 4, 8, 8)
        a = u(torch.cat([img, torch.full((1, 4, 8, 8), -1.0)], 1))
        b = u(torch.cat([img, torch.randn(1, 4, 8, 8)], 1))
        torch.testing.assert_close(a, b)

    def test_unfreeze_marks_conv_in_trainable(self):
        u = TinyUNet()
        expand_unet_conv_in(u, 4, zero_init=True)
        for p in u.parameters():
            p.requires_grad_(False)
        n = unfreeze_conv_in(u, 8)
        self.assertGreater(n, 0)
        self.assertTrue(all(p.requires_grad for p in u.conv_in.parameters()))
        # and the optimizer's filter would now pick it up
        self.assertGreater(len([p for p in u.parameters() if p.requires_grad]), 0)

    def test_unfreeze_rejects_an_unexpanded_conv(self):
        # Guards against the expansion silently not running, which would leave a 4-channel
        # conv fed 8 channels and fail deep inside the forward pass instead of here.
        with self.assertRaises(ValueError):
            unfreeze_conv_in(TinyUNet(), 8)

    def test_trained_slices_change_the_output(self):
        u = TinyUNet()
        expand_unet_conv_in(u, 4, zero_init=True)
        unfreeze_conv_in(u, 8)
        with torch.no_grad():  # stand in for what training would do
            u.conv_in.weight[:, 4:] = torch.randn_like(u.conv_in.weight[:, 4:]) * 0.1
        self.assertGreater(extra_channel_energy(u, 4), 0.0)
        img = torch.randn(1, 4, 8, 8)
        a = u(torch.cat([img, torch.full((1, 4, 8, 8), -1.0)], 1))
        b = u(torch.cat([img, torch.randn(1, 4, 8, 8)], 1))
        self.assertGreater((a - b).abs().max().item(), 1e-6)


class TestConvInRoundTrip(unittest.TestCase):
    def test_save_and_load_restores_the_weights(self):
        import tempfile

        src = TinyUNet()
        expand_unet_conv_in(src, 4, zero_init=True)
        with torch.no_grad():
            src.conv_in.weight[:, 4:] = torch.randn_like(src.conv_in.weight[:, 4:])
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(save_conv_in(src, d).is_file())
            # evaluation starts from the base model, four channels wide
            dst = TinyUNet()
            self.assertEqual(dst.conv_in.in_channels, 4)
            self.assertTrue(load_conv_in(dst, d))
            self.assertEqual(dst.conv_in.in_channels, 8)
            torch.testing.assert_close(dst.conv_in.weight, src.conv_in.weight)

    def test_missing_file_raises_by_default(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_conv_in(TinyUNet(), d)
            self.assertFalse(load_conv_in(TinyUNet(), d, strict=False))

    def test_filename_is_stable(self):
        self.assertEqual(FILENAME, "conv_in.safetensors")


class TestContourMaps(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel, mask):
        p = self.root / "masks" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, packed=np.packbits(mask.reshape(1, -1), axis=-1), n=1)

    def _image(self, rel, h, w):
        from PIL import Image

        p = self.root / "imgs" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(p)
        return p

    def test_contour_is_the_mask_boundary(self):
        m = np.zeros((20, 20), bool)
        m[5:15, 5:15] = True
        c = contour_of(m, 1)
        self.assertTrue(c[5, 10])       # on the edge
        self.assertFalse(c[10, 10])     # deep inside
        self.assertFalse(c[0, 0])       # far outside

    def test_map_is_resized_from_the_native_resolution(self):
        # Decoding straight into the requested shape reads the wrong bit count and returns
        # an empty map instead of erroring - the second failure this path had.
        m = np.zeros((40, 60), bool)
        m[10:30, 15:45] = True
        self._write("scene/frame_seg.npz", m)
        img = self._image("scene/frame.png", 40, 60)

        cache = ContourCache(self.root / "masks", 1)
        native = cache.contour_for(img, self.root / "imgs", (40, 60))
        resized = cache.contour_for(img, self.root / "imgs", (20, 30))
        self.assertGreater((native > 0).mean(), 0)
        self.assertGreater((resized > 0).mean(), 0)
        self.assertLess(abs((native > 0).mean() - (resized > 0).mean()), 0.02)

    def test_values_are_the_training_range(self):
        m = np.zeros((16, 16), bool)
        m[4:12, 4:12] = True
        self._write("s/f_seg.npz", m)
        img = self._image("s/f.png", 16, 16)
        c = ContourCache(self.root / "masks", 1).contour_for(img, self.root / "imgs", (16, 16))
        self.assertEqual(set(np.unique(c).tolist()), {-1.0, 1.0})

    def test_missing_mask_gives_an_empty_map(self):
        img = self._image("s/none.png", 16, 16)
        c = ContourCache(self.root / "masks", 1).contour_for(img, self.root / "imgs", (16, 16))
        self.assertTrue((c == -1.0).all())

    def test_guard_rejects_a_cache_of_empty_masks(self):
        # build_sam_masks writes n=0 for frames without detections, so existence alone
        # passes even when every contour is blank.
        for i in range(10):
            p = self.root / "masks" / "s" / f"f{i}_seg.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, packed=np.zeros((0, 0), np.uint8), n=0)
        imgs = [self._image(f"s/f{i}.png", 16, 16) for i in range(10)]
        with self.assertRaises(RuntimeError):
            ContourCache(self.root / "masks", 1).check_root(imgs, self.root / "imgs", None)

    def test_batch_modes_differ(self):
        for i in range(3):
            m = np.zeros((16, 16), bool)
            m[2 + i : 8 + i, 2 + i : 8 + i] = True
            self._write(f"s/f{i}_seg.npz", m)
        imgs = [str(self._image(f"s/f{i}.png", 16, 16)) for i in range(3)]
        cache = ContourCache(self.root / "masks", 1)
        real = cache.batch(imgs, self.root / "imgs", 3, (16, 16), "cpu", torch.float32)
        shuf = cache.batch(imgs, self.root / "imgs", 3, (16, 16), "cpu", torch.float32,
                           shuffle=imgs[1:] + imgs[:1])
        zero = cache.batch(imgs, self.root / "imgs", 3, (16, 16), "cpu", torch.float32,
                           shuffle=[])
        self.assertFalse(torch.equal(real, shuf))
        self.assertTrue((zero == -1.0).all())
        self.assertGreater((real > 0).float().mean().item(), 0)


if __name__ == "__main__":
    unittest.main()
