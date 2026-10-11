#!/bin/sh
# Fetch one of Marigold's evaluation archives, resuming if interrupted.
#
#   ./fetch_eval_data.sh diode/diode_val.tar
#   ./fetch_eval_data.sh eth3d/eth3d.tar
#
# Written after three attempts to assess ETH3D by streaming or seeking into the tar
# without downloading it. All three failed, and all three cost more effort than the
# download costs: 46 GB is wall-clock time in the background, not work.
set -e
[ -n "$1" ] || { echo "usage: $0 <subdir>/<file>.tar"; exit 2; }
SPEC="$1"
DEST="D:/lotus/data/marigold_depth_eval/${SPEC%%/*}"
mkdir -p "$DEST"
cd /d/lotus/marigold-v2
./.venv/Scripts/python.exe - "$SPEC" "$DEST/${SPEC##*/}" <<'PY'
import sys, time
from pathlib import Path
import httpx
BASE = "https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset"
url, dst = f"{BASE}/{sys.argv[1]}", Path(sys.argv[2])
have = dst.stat().st_size if dst.is_file() else 0
total = int(httpx.head(url, timeout=60, follow_redirects=True).headers["content-length"])
if have == total:
    print(f"already complete: {total/2**30:.2f} GB"); raise SystemExit
print(f"resuming at {have/2**30:.2f} / {total/2**30:.2f} GB" if have
      else f"downloading {total/2**30:.2f} GB")
t0, got, last = time.time(), have, have
with httpx.stream("GET", url, headers={"Range": f"bytes={have}-"} if have else {},
                  timeout=600, follow_redirects=True) as r:
    r.raise_for_status()
    with open(dst, "ab" if have else "wb") as f:
        for chunk in r.iter_bytes(chunk_size=8 << 20):
            f.write(chunk); got += len(chunk)
            if got - last >= (2 << 30):
                last = got
                mb_s = (got - have) / 2**20 / max(time.time() - t0, 1)
                eta = (total - got) / 2**20 / max(mb_s, 1e-6) / 60
                print(f"  {got/2**30:.1f} / {total/2**30:.1f} GB  "
                      f"{mb_s:.1f} MB/s  eta {eta:.0f} min", flush=True)
print(f"done: {dst.stat().st_size/2**30:.2f} GB")
PY
