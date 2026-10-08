"""Where a frame's cached artefacts live.

Re-exported from eval_object_oracle_ceiling rather than redefined: twenty-two scripts
already import `_cache_path` from there and every cached prediction and SAM mask on disk
was written through it, so a second implementation here could only ever drift from the
layout the caches actually have.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_object_oracle_ceiling import _cache_path as cache_path

__all__ = ["cache_path"]
