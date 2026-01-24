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
FALLBACK_SRC = Path.cwd() / "src"
if str(ROOT) not in sys.path:  # pragma: no cover - import path fix
    sys.path.insert(0, str(ROOT))
if SRC.exists() and str(SRC) not in sys.path:  # pragma: no cover - import path fix
    sys.path.insert(0, str(SRC))
if FALLBACK_SRC.exists() and str(FALLBACK_SRC) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(FALLBACK_SRC))

def _load_app_main():
    try:
        from pharma_financial.app import main
    except ModuleNotFoundError as exc:  # pragma: no cover - executed when deps missing
        if exc.name in {"pharma_financial", "pharma_financial.app"}:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            if str(FALLBACK_SRC) not in sys.path:
                sys.path.insert(0, str(FALLBACK_SRC))
            try:
                from pharma_financial.app import main  # type: ignore[redefined-outer-name]
            except Exception as retry_exc:
                raise SystemExit(
                    "Unable to import `pharma_financial`. Ensure the `src` directory "
                    "is present alongside streamlit_app.py or install the package with "
                    "`pip install -e .`."
                ) from retry_exc
            return main
        if exc.name == "streamlit":
            raise SystemExit(
                "Streamlit is not installed. Install project dependencies with "
                "`pip install -r requirements.txt` before running the app."
            ) from exc
        raise SystemExit(
            f"Unable to import required dependency '{exc.name}'. Ensure the dependency is "
            "installed or add it to requirements.txt."
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected import failure
        raise SystemExit(
            "Unable to import the Streamlit application. "
            "Check dependency installation and module paths."
        ) from exc
    return main


main = _load_app_main()


def _running_with_streamlit() -> bool:
    """Return ``True`` when executed via ``streamlit run``.

    When the file is executed directly with ``python streamlit_app.py`` the Streamlit
    runtime is not initialised which leads to ``st.session_state`` access raising the
    exception reported by the user. Detect that situation early and exit with a clear
    guidance message instead of letting the import stack fail deeper inside the app.
    """

    try:  # pragma: no cover - depends on Streamlit internals
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover - older Streamlit versions
        get_script_run_ctx = None  # type: ignore[assignment]

    if get_script_run_ctx is None:
        try:  # pragma: no cover - Streamlit < 1.18
            from streamlit.scriptrunner import get_script_run_ctx
        except Exception:
            return False

    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if _running_with_streamlit():
        main()
    else:  # pragma: no cover - executed only when run without ``streamlit run``
        sys.stderr.write(
            "This module is a Streamlit application. Launch it with "
            "`streamlit run streamlit_app.py` instead of `python streamlit_app.py`.\n"
        )
        sys.stderr.flush()
