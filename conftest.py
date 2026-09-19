"""Make `src/` importable without an install step.

The runtime is deliberately installable-free: no build backend, no editable
install, no network. Tests and scripts put `src/` on the path explicitly.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
