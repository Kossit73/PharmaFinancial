"""Entry point for deploying the pharmaceuticals model on Streamlit Cloud."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the ``src`` directory is importable when the package has not been
# installed. Streamlit executes this file directly and the execution working
# directory may not include ``src`` on ``PYTHONPATH``. Adding it explicitly keeps
# ``from financial_models import ...`` imports functional both locally and on
# Streamlit Cloud deployments without requiring an editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:  # pragma: no cover - import path fix
    sys.path.insert(0, str(SRC))

try:
    from financial_models.pharma_app import main
except ModuleNotFoundError as exc:  # pragma: no cover - executed when deps missing
    if exc.name == "streamlit":
        raise SystemExit(
            "Streamlit is not installed. Install project dependencies with "
            "`pip install -r requirements.txt` before running the app."
        ) from exc
    raise


def _running_with_streamlit() -> bool:
    """Return ``True`` when executed via ``streamlit run``.

    When the file is executed directly with ``python pharma_streamlit_app.py`` the Streamlit
    runtime is not initialised which leads to ``st.session_state`` access raising the
    exception reported by the user. Detect that situation early and exit with a clear
    guidance message instead of letting the import stack fail deeper inside the app.
    """

    try:  # pragma: no cover - depends on Streamlit internals
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover - older Streamlit versions
        return False

    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if _running_with_streamlit():
        main()
    else:  # pragma: no cover - executed only when run without ``streamlit run``
        sys.stderr.write(
            "This module is a Streamlit application. Launch it with "
            "`streamlit run pharma_streamlit_app.py` instead of `python pharma_streamlit_app.py`.\n"
        )
        sys.stderr.flush()
