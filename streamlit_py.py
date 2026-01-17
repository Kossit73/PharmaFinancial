"""Streamlit entry point for deployments expecting `streamlit_py.py`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pharma_financial.app import main


if __name__ == "__main__":  # pragma: no cover - Streamlit executes the script directly
    main()
