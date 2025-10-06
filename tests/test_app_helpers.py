import importlib
import sys
import types
import unittest


class _NoOp:
    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive noop
        return None

    def __getattr__(self, _name):  # pragma: no cover - defensive noop
        return self


class DummyStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self._no_op = _NoOp()
        self.calls: list[str] = []

    def rerun(self):  # pragma: no cover - behaviour exercised in tests
        self.calls.append("rerun")
        raise RuntimeError("rerun not available")

    def experimental_rerun(self):  # pragma: no cover - exercised in tests
        self.calls.append("experimental_rerun")
        raise RuntimeError("experimental rerun not available")

    def __getattr__(self, name):  # pragma: no cover - fallback for unused attrs
        return getattr(self._no_op, name, self._no_op)


class RerunHelperTest(unittest.TestCase):
    def setUp(self):
        self.original_streamlit = sys.modules.get("streamlit")
        self.original_runtime = sys.modules.get("streamlit.runtime")

        stub = DummyStreamlit()
        runtime = types.ModuleType("streamlit.runtime")
        runtime.exists = lambda: False  # type: ignore[attr-defined]

        sys.modules["streamlit"] = stub
        sys.modules["streamlit.runtime"] = runtime

        if "pharma_financial.app" in sys.modules:
            del sys.modules["pharma_financial.app"]

        importlib.invalidate_caches()
        self.stub = stub
        self.app = importlib.import_module("pharma_financial.app")

    def tearDown(self):
        if self.original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = self.original_streamlit

        if self.original_runtime is None:
            sys.modules.pop("streamlit.runtime", None)
        else:
            sys.modules["streamlit.runtime"] = self.original_runtime

        if "pharma_financial.app" in sys.modules:
            del sys.modules["pharma_financial.app"]

    def test_rerun_helper_handles_missing_runtime(self):
        self.app._rerun()
        self.assertEqual(self.stub.calls, ["rerun", "experimental_rerun"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

