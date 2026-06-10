#!/usr/bin/env python3
"""Launcher for Mr. House.

Adds ``src`` to the path (so you can run without installing) and delegates to
``mr_house.main``.

    python run.py            # full experience (voice + CRT display)
    python run.py --check     # report which subsystems are available
    python run.py --no-display
    python run.py --fullscreen # start in fullscreen (toggle live with F11)
    python run.py --text      # type questions to test the brain
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mr_house.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

