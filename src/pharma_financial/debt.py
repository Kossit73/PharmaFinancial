"""Debt amortisation helpers shared by the UI and financial engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

from .inputs import DebtEntry


@dataclass
class DebtPeriod:
    """Represents a single period within a debt amortisation schedule."""

    year: int
    interest: float
    principal: float
    payment: float
    outstanding: float


def amortise_entry(
    entry: DebtEntry,
    rate: float,
    years: Sequence[int],
    year_to_index: Mapping[int, int] | None = None,
) -> List[DebtPeriod]:
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

    if year_to_index is None:
        try:
            start_index = years.index(entry.year)
        except ValueError:
            return schedule
    else:
        start_index = year_to_index.get(entry.year)
        if start_index is None:
            return schedule

    principal = float(entry.amount)
    if principal <= 0.0:
        return schedule

    opening_outstanding = float(entry.outstanding or principal)
    opening_outstanding = min(opening_outstanding, principal)
    balance = max(opening_outstanding, 0.0)
    duration = max(int(entry.duration or 0), 1)

    for offset in range(duration):
        idx = start_index + offset
        if idx >= len(years):
            break

        if balance <= 0.0:
            break

        remaining_periods = duration - offset
        remaining_periods = max(remaining_periods, 1)

        interest_payment = balance * float(rate)

        principal_payment = balance / remaining_periods
        if principal_payment > balance:
            principal_payment = balance

        total_payment = interest_payment + principal_payment

        balance = max(balance - principal_payment, 0.0)

        schedule.append(
            DebtPeriod(
                year=years[idx],
                interest=interest_payment,
                principal=principal_payment,
                payment=total_payment,
                outstanding=balance,
            )
        )

    return schedule


def amortise_entries(
    entries: Iterable[DebtEntry], rate: float, years: Sequence[int]
) -> Tuple[List[float], List[float], List[float], List[List[DebtPeriod]]]:
    """Aggregate amortisation schedules for multiple debt entries."""

    horizon = len(years)
    interest_schedule = [0.0 for _ in range(horizon)]
    principal_schedule = [0.0 for _ in range(horizon)]
    outstanding_schedule = [0.0 for _ in range(horizon)]
    entry_schedules: List[List[DebtPeriod]] = []

    year_to_index = {year: idx for idx, year in enumerate(years)}
    for entry in entries:
        schedule = amortise_entry(entry, rate, years, year_to_index)
        entry_schedules.append(schedule)
        for period in schedule:
            idx = year_to_index.get(period.year)
            if idx is None:
                continue
            interest_schedule[idx] += period.interest
            principal_schedule[idx] += period.principal
            outstanding_schedule[idx] += period.outstanding

    return interest_schedule, principal_schedule, outstanding_schedule, entry_schedules
