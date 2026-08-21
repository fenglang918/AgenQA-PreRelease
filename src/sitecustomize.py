"""Bootstrap imports for `python src/...` entrypoints.

Running `python src/cli.py` puts `src/` on `sys.path`, but not necessarily the
repo root. The root is needed now that `infra/` is a top-level package.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
