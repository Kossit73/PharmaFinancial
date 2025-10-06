"""Entry point for deploying the financial model on Streamlit Cloud."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ``src`` directory is importable when the package has not been
# installed. Streamlit executes this file directly and the execution working
# directory may not include ``src`` on ``PYTHONPATH``. Adding it explicitly keeps
# ``from pharma_financial import ...`` imports functional both locally and on
# Streamlit Cloud deployments without requiring an editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:  # pragma: no cover - import path fix
    sys.path.insert(0, str(SRC))

try:
    from pharma_financial.app import main
except ModuleNotFoundError as exc:  # pragma: no cover - executed when deps missing
    if exc.name == "streamlit":
        raise SystemExit(
            "Streamlit is not installed. Install project dependencies with "
            "`pip install -r requirements.txt` before running the app."
        ) from exc
    raise


if __name__ == "__main__":
    main()
