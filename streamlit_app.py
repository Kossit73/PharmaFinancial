"""Entry point for deploying the financial model on Streamlit Cloud."""

from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from pharma_financial.app import main


if __name__ == "__main__":
    main()
