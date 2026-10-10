"""Measure ETH3D's ground-truth density without downloading 46 GB.

The last of the five standard benchmarks. KITTI failed on density (11.2% of adjacent
pairs), DIODE on the protocol that removes 99.4% of depth discontinuities. ETH3D has no
such filter in Marigold's evaluation code, so density is the open question.

ETH3D's depth files are raw float32 blobs named .JPG, read at the RGB's resolution -
which is why this has to open an image to know the shape.
"""
import io
import tarfile

import httpx
import numpy as np
from PIL import Image

URL = ("https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/"
       "eth3d/eth3d.tar")
CAP = 700 * 2 ** 20
WANT = 6

rgb_shapes, depth_blobs, rows = {}, {}, []
read = 0


def key(name):
    return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]


with httpx.stream("GET", URL, timeout=180, follow_redirects=True) as r:
    r.raise_for_status()
    chunks = r.iter_bytes(chunk_size=1 << 20)

    class S(io.RawIOBase):
        def __init__(self):
            self.buf = b""
        def readable(self):
            return True
        def readinto(self, b):
            global read
            while len(self.buf) < len(b):
                try:
                    c = next(chunks)
                except StopIteration:
                    break
                read += len(c)
                self.buf += c
                if read > CAP:
                    break
            n = min(len(b), len(self.buf))
            b[:n] = self.buf[:n]
            self.buf = self.buf[n:]
            return n

    with tarfile.open(fileobj=io.BufferedReader(S(), buffer_size=1 << 20),
                      mode="r|") as tf:
        for m in tf:
            if read > CAP:
                break
            if not m.isfile():
                continue
            data = tf.extractfile(m).read()
            k = key(m.name)
            if "/rgb/" in m.name or m.name.startswith("rgb/"):
                try:
                    rgb_shapes[k] = Image.open(io.BytesIO(data)).size[::-1]
                except Exception:
                    pass
            elif "ground_truth_depth" in m.name or "/depth/" in m.name:
                depth_blobs[k] = data
            for kk in set(rgb_shapes) & set(depth_blobs):
                h, w = rgb_shapes.pop(kk)
                blob = depth_blobs.pop(kk)
                if len(blob) != h * w * 4:
                    rows.append((kk, (h, w), None, None, len(blob)))
                    continue
                d = np.frombuffer(blob, np.float32).reshape(h, w).astype(np.float64)
                v = np.isfinite(d) & (d > 0.01)
                pair = ((v[:, :-1] & v[:, 1:]).mean() + (v[:-1, :] & v[1:, :]).mean()) / 2
                rows.append((kk, (h, w), v.mean(), pair, len(blob)))
            if len(rows) >= WANT:
                break

print(f"read {read / 2**20:.0f} MB of {46.04 * 1024:.0f} MB")
for k, shape, v, p, nbytes in rows:
    if v is None:
        print(f"  {k:22s} {str(shape):>14}  blob {nbytes} bytes - not h*w*4, format unknown")
    else:
        print(f"  {k:22s} {str(shape):>14}  valid {v*100:6.2f}%   both-valid pairs {p*100:6.2f}%")
ok = [r for r in rows if r[2] is not None]
if ok:
    print(f"\nmean valid {np.mean([r[2] for r in ok])*100:.2f}%   "
          f"mean pairs {np.mean([r[3] for r in ok])*100:.2f}%")
print("reference: NYUv2 ~100% / DIODE 99.7% indoor / KITTI 11.2% pairs")
