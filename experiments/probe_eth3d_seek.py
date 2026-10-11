"""Find ETH3D's depth files by seeking into the tar instead of streaming to them.

Two streaming attempts read 704 MB and 908 MB without reaching a single depth file: the
archive lists all of `rgb/` first, and at ~9 MB a JPEG that prefix is many gigabytes.

tar has no index, but it does not need one here. Every member is preceded by a 512-byte
header carrying the name at offset 0 and the magic "ustar" at offset 257, and headers sit
on 512-byte boundaries. So a 2 MB window read with an HTTP Range request anywhere in the
file will contain several headers, and reading a handful of windows maps the archive
cheaply enough to find where the depth section starts and what one of those files looks
like.
"""
import httpx
import numpy as np

URL = ("https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/"
       "eth3d/eth3d.tar")
WINDOW = 2 << 20


def headers_in(buf, base):
    """(offset, name, size) for every tar header in `buf`."""
    out = []
    for off in range(0, len(buf) - 512, 512):
        blk = buf[off:off + 512]
        if blk[257:262] != b"ustar":
            continue
        name = blk[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        raw = blk[124:136].split(b"\0", 1)[0].strip()
        try:
            size = int(raw, 8) if raw else 0
        except ValueError:
            size = 0
        if name:
            out.append((base + off, name, size))
    return out


with httpx.Client(timeout=120, follow_redirects=True) as c:
    total = int(c.head(URL).headers["content-length"])
    print(f"archive {total / 2**30:.2f} GB")

    print("\n--- map ---")
    found = None
    for frac in [i / 10 for i in range(11)]:
        start = min(int(total * frac), total - WINDOW)
        start -= start % 512
        buf = c.get(URL, headers={"Range": f"bytes={start}-{start + WINDOW - 1}"}).content
        hs = headers_in(buf, start)
        names = [h[1] for h in hs]
        kinds = sorted({n.split("/")[1] for n in names if n.count("/") > 1})
        print(f"  {frac * 100:3.0f}%  {len(hs):3d} headers  {kinds}")
        if found is None:
            for off, name, size in hs:
                if "ground_truth_depth" in name and size > 0:
                    found = (off, name, size)
                    break

    if not found:
        print("\nno depth header landed in a window; archive mapped above")
        raise SystemExit

    off, name, size = found
    print(f"\n--- reading {name}  ({size / 2**20:.1f} MB) ---")
    s = off + 512
    blob = c.get(URL, headers={"Range": f"bytes={s}-{s + size - 1}"}).content
    a = np.frombuffer(blob, np.float32)
    v = np.isfinite(a) & (a > 0.01)
    shape = next(((h, w) for h, w in
                  [(4032, 6048), (6048, 4032), (3024, 4032), (4032, 3024),
                   (1360, 2048), (2048, 1360)] if h * w == a.size), None)
    print(f"  {a.size} values, shape {shape}")
    print(f"  valid {v.mean() * 100:.2f}%")
    if shape:
        vv = v.reshape(shape)
        pair = ((vv[:, :-1] & vv[:, 1:]).mean() + (vv[:-1, :] & vv[1:, :]).mean()) / 2
        print(f"  both-valid adjacent pairs {pair * 100:.2f}%")
    print("  reference: NYUv2 ~100% / DIODE 99.7% indoor / KITTI 11.2% pairs")
