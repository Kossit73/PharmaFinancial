"""Debt amortisation helpers shared by the UI and financial engine."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Iterable, List, Sequence, Tuple

from .inputs import DebtEntry


@dataclass
class DebtPeriod:
    """Represents a single period within a debt amortisation schedule."""

    year: int
    payment: float
    cumulative: float
    outstanding: float


def amortise_entry(entry: DebtEntry, rate: float, years: Sequence[int]) -> List[DebtPeriod]:
    """Generate an amortisation schedule for a single debt entry.

    Parameters
    ----------
    entry:
        The configured debt item containing the year, amount, outstanding
        balance, and duration.
    rate:
        Annual interest rate applied to the outstanding balance.
    years:
        Projection horizon used to align the generated schedule.
    """

    schedule: List[DebtPeriod] = []
    if not years:
        return schedule

    try:
        start_index = years.index(entry.year)
    except ValueError:
        return schedule

    principal = max(float(entry.amount), float(entry.outstanding))
    opening_outstanding = min(float(entry.outstanding), principal)
    cumulative_paid = principal - opening_outstanding
    duration = max(int(entry.duration or 0), 1)

    for offset in range(duration):
        idx = start_index + offset
        if idx >= len(years):
            break

        current_outstanding = max(principal - cumulative_paid, 0.0)
        if current_outstanding <= 0.0:
            break

        remaining_periods = duration - offset
        principal_share = (
            current_outstanding / remaining_periods
            if remaining_periods > 0
            else current_outstanding
        )
        payment = max(current_outstanding * float(rate), principal_share)
        if payment > current_outstanding:
            payment = current_outstanding

        cumulative_paid += payment
        outstanding_after = max(principal - cumulative_paid, 0.0)

        schedule.append(
            DebtPeriod(
                year=years[idx],
                payment=payment,
                cumulative=cumulative_paid,
                outstanding=outstanding_after,
            )
        )

    return schedule


def amortise_entries(
    entries: Iterable[DebtEntry], rate: float, years: Sequence[int]
) -> Tuple[List[float], List[float], List[float], List[List[DebtPeriod]]]:
    """Aggregate amortisation schedules for multiple debt entries."""

    horizon = len(years)
    interest_schedule = [0.0 for _ in range(horizon)]
    outstanding_schedule = [0.0 for _ in range(horizon)]
    entry_schedules: List[List[DebtPeriod]] = []

    for entry in entries:
        schedule = amortise_entry(entry, rate, years)
        entry_schedules.append(schedule)
        for period in schedule:
            try:
                idx = years.index(period.year)
            except ValueError:
                continue
            interest_schedule[idx] += period.payment
            outstanding_schedule[idx] += period.outstanding

    cumulative_schedule = list(accumulate(interest_schedule)) if horizon else []
    return interest_schedule, outstanding_schedule, cumulative_schedule, entry_schedules
