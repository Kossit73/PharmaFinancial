import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_models.pharma.debt import amortise_entries, amortise_entry
from financial_models.pharma.inputs import DebtEntry


def test_amortise_entry_generates_full_schedule():
    entry = DebtEntry(year=2024, amount=100.0, outstanding=100.0, duration=5)
    schedule = amortise_entry(entry, rate=0.1, years=list(range(2024, 2034)))

    assert len(schedule) == 5
    principals = [round(period.principal, 2) for period in schedule]
    interests = [round(period.interest, 2) for period in schedule]
    assert principals == [20.0, 20.0, 20.0, 20.0, 20.0]
    assert interests == [10.0, 8.0, 6.0, 4.0, 2.0]
    assert schedule[-1].outstanding == 0.0


def test_amortise_entries_aggregates_interest_and_outstanding():
    years = list(range(2024, 2034))
    entries = [
        DebtEntry(year=2024, amount=100.0, outstanding=100.0, duration=5),
        DebtEntry(year=2026, amount=50.0, outstanding=50.0, duration=3),
    ]

    interest, principal, outstanding, schedules = amortise_entries(entries, 0.1, years)

    assert len(schedules) == 2
    assert round(sum(principal), 2) == 150.0
    first_year_index = years.index(2024)
    assert round(interest[first_year_index], 2) == 10.0
    assert round(outstanding[first_year_index], 2) == 80.0
    final_year_index = years.index(2028)
    assert outstanding[final_year_index] == 0.0
