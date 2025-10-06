"""Module entry point for running the CLI."""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - exercised via ``python -m`` in deployment environments
    from .cli import main
except ImportError:  # pragma: no cover - fallback when package not installed
    # When the package has not been installed (e.g., running directly from the
    # repository without ``pip install -e .``), ensure the parent ``src``
    # directory is available on ``sys.path`` so absolute imports work.
    PACKAGE_ROOT = Path(__file__).resolve().parent
    SRC_ROOT = PACKAGE_ROOT.parent
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from pharma_financial.cli import main

if __name__ == "__main__":
    main()
