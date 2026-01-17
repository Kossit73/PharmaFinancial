"""Streamlit entry point for deployments expecting `streamlit_py.py`."""
from pharma_financial.app import main


if __name__ == "__main__":  # pragma: no cover - Streamlit executes the script directly
    main()
