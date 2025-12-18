import unittest

from financial_models.biotech import ValuationEngine, build_portfolio, load_inputs


class BiotechModelTest(unittest.TestCase):
    def test_biotech_model_runs_with_defaults(self) -> None:
        inputs = load_inputs()
        portfolio = build_portfolio(inputs)
        result = ValuationEngine(portfolio).run()
        # basic sanity checks
        self.assertGreater(result.rnpv, 0)
        self.assertFalse(result.consolidated.empty)
        self.assertIn("revenue", result.consolidated.columns)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
