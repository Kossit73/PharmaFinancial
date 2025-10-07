import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pharma_financial.debt import amortise_entries, amortise_entry
from pharma_financial.inputs import DebtEntry


def test_amortise_entry_generates_full_schedule():
    entry = DebtEntry(year=2024, amount=100.0, outstanding=100.0, duration=5)
    schedule = amortise_entry(entry, rate=0.1, years=list(range(2024, 2034)))

    assert len(schedule) == 5
    payments = [round(period.payment, 2) for period in schedule]
    assert payments == [20.0, 20.0, 20.0, 20.0, 20.0]
    assert round(schedule[-1].cumulative, 2) == 100.0
    assert schedule[-1].outstanding == 0.0


def test_amortise_entries_aggregates_interest_and_outstanding():
    years = list(range(2024, 2034))
    entries = [
        DebtEntry(year=2024, amount=100.0, outstanding=100.0, duration=5),
        DebtEntry(year=2026, amount=50.0, outstanding=50.0, duration=3),
    ]

    interest, outstanding, cumulative, schedules = amortise_entries(entries, 0.1, years)

    assert len(schedules) == 2
    assert round(sum(interest), 2) == 150.0
    assert round(cumulative[-1], 2) == round(sum(interest), 2)
    first_year_index = years.index(2024)
    assert round(outstanding[first_year_index], 2) == 80.0
    final_year_index = years.index(2028)
    assert outstanding[final_year_index] == 0.0
