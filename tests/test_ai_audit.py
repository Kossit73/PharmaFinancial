import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_financial.ai import GenerativeAdvisor
from pharma_financial.inputs import AIParameters
from pharma_financial.table import build_table


class GenerativeAdvisorAuditTest(unittest.TestCase):
    def setUp(self):
        self.summary = build_table(
            ["NPV", "IRR", "Payback Period"],
            {"Value": [10_000_000.0, 0.18, 5.0]},
            index_name="Metric",
        )
        self.income = build_table(
            [2029, 2030],
            {
                "Net Revenue": [90_000_000.0, 100_000_000.0],
                "EBITDA": [15_000_000.0, 17_000_000.0],
                "Net Income": [8_000_000.0, 9_500_000.0],
            },
        )
        self.cash_flow = build_table(
            [2029, 2030],
            {"Net Change in Cash": [1_000_000.0, 1_500_000.0], "Ending Cash": [12_000_000.0, 13_500_000.0]},
        )

    def test_model_response_missing_domains_falls_back(self):
        params = AIParameters(enabled=True, provider="OpenAI", api_key="test-key")
        advisor = GenerativeAdvisor(params)
        advisor._invoke_model = lambda _prompt: "Revenue and margin improved with better profitability."

        text = advisor.summarise(
            summary=self.summary,
            income=self.income,
            cash_flow=self.cash_flow,
            ml_table=None,
        )

        self.assertIn("Pharmaceutical management practice review", text)
        self.assertEqual(advisor.metadata.get("pharma_management_audit_status"), "failed")

    def test_model_response_covering_domains_is_accepted(self):
        params = AIParameters(enabled=True, provider="OpenAI", api_key="test-key")
        advisor = GenerativeAdvisor(params)
        advisor._invoke_model = lambda _prompt: (
            "Patient safety remains central, GMP quality controls are maintained, "
            "regulatory compliance milestones are on track, and supply continuity risk is monitored."
        )

        text = advisor.summarise(
            summary=self.summary,
            income=self.income,
            cash_flow=self.cash_flow,
            ml_table=None,
        )

        self.assertIn("patient safety", text.lower())
        self.assertEqual(advisor.metadata.get("pharma_management_audit_status"), "passed")
        self.assertEqual(advisor.metadata.get("status"), "model_response")


if __name__ == "__main__":
    unittest.main()
