"""Measure DIODE's ground-truth density without downloading all 5.96 GB.

tar is sequential, so streaming it and stopping once a handful of depth/mask pairs have
gone past answers the only question that decides whether DIODE is worth the download:
can a boundary metric be computed on it at all. BF1 compares NEIGHBOURING pixels, so what
matters is not the fraction of valid pixels but the fraction of adjacent pairs where both
are valid - on KITTI that is 11.2% against NYUv2's ~100%, which is why KITTI is a poor
fit despite being a standard benchmark.
"""
import io
import tarfile

import httpx
import numpy as np

URL = ("https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/"
       "diode/diode_val.tar")
CAP_BYTES = 400 * 2 ** 20
WANT = 12

depths, masks = {}, {}
rows = []
read = 0

with httpx.stream("GET", URL, timeout=120, follow_redirects=True) as r:
    r.raise_for_status()
    raw = r.iter_bytes(chunk_size=1 << 20)

    class Counting(io.RawIOBase):
        def __init__(self):
            self.buf = b""
        def readable(self):
            return True
        def readinto(self, b):
            global read
            while len(self.buf) < len(b):
                try:
                    chunk = next(raw)
                except StopIteration:
                    break
                read += len(chunk)
                self.buf += chunk
                if read > CAP_BYTES:
                    break
            n = min(len(b), len(self.buf))
            b[:n] = self.buf[:n]
            self.buf = self.buf[n:]
            return n

    stream = io.BufferedReader(Counting(), buffer_size=1 << 20)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or read > CAP_BYTES:
                if read > CAP_BYTES:
                    break
                continue
            name = m.name
            if name.endswith("_depth.npy"):
                depths[name[: -len("_depth.npy")]] = np.load(
                    io.BytesIO(tf.extractfile(m).read()))
            elif name.endswith("_depth_mask.npy"):
                masks[name[: -len("_depth_mask.npy")]] = np.load(
                    io.BytesIO(tf.extractfile(m).read()))
            for k in set(depths) & set(masks):
                d = np.squeeze(depths.pop(k)).astype(np.float64)
                mk = np.squeeze(masks.pop(k)).astype(bool)
                v = mk & np.isfinite(d) & (d > 1e-3)
                pair_h = (v[:, :-1] & v[:, 1:]).mean()
                pair_v = (v[:-1, :] & v[1:, :]).mean()
                rows.append((k.split("/")[-1], d.shape, v.mean(),
                             (pair_h + pair_v) / 2))
            if len(rows) >= WANT:
                break

print(f"read {read / 2**20:.0f} MB of {5.96 * 1024:.0f} MB")
print(f"{'frame':34s} {'shape':>12} {'valid':>8} {'both-valid pairs':>18}")
for name, shape, v, p in rows:
    print(f"{name[:34]:34s} {str(shape):>12} {v * 100:7.1f}% {p * 100:17.1f}%")
if rows:
    print(f"\nmean valid {np.mean([r[2] for r in rows]) * 100:.1f}%   "
          f"mean both-valid pairs {np.mean([r[3] for r in rows]) * 100:.1f}%")
    print("reference: NYUv2 ~100%, KITTI 20.2% valid / 11.2% pairs")
