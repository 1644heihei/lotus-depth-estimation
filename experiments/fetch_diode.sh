#!/bin/sh
# DIODE val (5.96 GB). The only standard benchmark left whose ground truth is dense
# enough for a boundary metric: 75.8% of adjacent pixel pairs are both valid, against
# KITTI's 11.2%. Resumes if interrupted.
set -e
D="D:/lotus/data/marigold_depth_eval/diode"
mkdir -p "$D"
cd /d/lotus/marigold-v2
./.venv/Scripts/python.exe - "$D/diode_val.tar" <<'PY'
import sys
from pathlib import Path
import httpx
URL = ("https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/"
       "diode/diode_val.tar")
dst = Path(sys.argv[1])
have = dst.stat().st_size if dst.is_file() else 0
total = int(httpx.head(URL, timeout=60, follow_redirects=True).headers["content-length"])
if have == total:
    print(f"already complete: {total/2**30:.2f} GB"); raise SystemExit
headers = {"Range": f"bytes={have}-"} if have else {}
print(f"resuming at {have/2**30:.2f} / {total/2**30:.2f} GB" if have else
      f"downloading {total/2**30:.2f} GB")
with httpx.stream("GET", URL, headers=headers, timeout=300, follow_redirects=True) as r:
    r.raise_for_status()
    got = have
    with open(dst, "ab" if have else "wb") as f:
        for chunk in r.iter_bytes(chunk_size=4 << 20):
            f.write(chunk); got += len(chunk)
            if got % (512 << 20) < (4 << 20):
                print(f"  {got/2**30:.2f} / {total/2**30:.2f} GB", flush=True)
print(f"done: {dst.stat().st_size/2**30:.2f} GB")
PY
