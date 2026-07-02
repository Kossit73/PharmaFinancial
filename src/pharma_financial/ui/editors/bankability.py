"""Editors for bankability thresholds, evidence, and downside cases."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _editor_rows(rows: Any) -> list[dict[str, object]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        try:
            return list(rows.to_dict("records"))
        except Exception:
            pass
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def render_bankability_controls_section(payload: dict) -> None:
    from ... import app as legacy

    st = legacy.st
    config = payload.setdefault("bankability", {})
    st.subheader("Bankability Controls")
    cols = st.columns(4)
    config["min_irr"] = cols[0].number_input(
        "Minimum IRR",
        min_value=0.0,
        max_value=1.0,
        value=_float_value(config.get("min_irr"), 0.18),
        step=0.01,
        format="%.2f",
        key="bankability_min_irr",
    )
    config["min_dscr"] = cols[1].number_input(
        "Minimum DSCR",
        min_value=0.0,
        value=_float_value(config.get("min_dscr"), 1.25),
        step=0.05,
        format="%.2f",
        key="bankability_min_dscr",
    )
    config["max_discounted_payback"] = cols[2].number_input(
        "Max Discounted Payback",
        min_value=0.0,
        value=_float_value(config.get("max_discounted_payback"), 7.0),
        step=0.5,
        format="%.1f",
        key="bankability_max_payback",
    )
    config["min_cash_buffer"] = cols[3].number_input(
        "Minimum Cash Buffer",
        value=_float_value(config.get("min_cash_buffer"), 5.0),
        step=1.0,
        format="%.2f",
        key="bankability_min_cash",
    )

    cols = st.columns(3)
    config["min_assumption_quality"] = cols[0].number_input(
        "Min Assumption Quality",
        min_value=0.0,
        max_value=100.0,
        value=_float_value(config.get("min_assumption_quality"), 80.0),
        step=1.0,
        format="%.1f",
        key="bankability_min_quality",
    )
    config["min_evidence_coverage"] = cols[1].number_input(
        "Min Evidence Coverage",
        min_value=0.0,
        max_value=1.0,
        value=_float_value(config.get("min_evidence_coverage"), 0.75),
        step=0.05,
        format="%.2f",
        key="bankability_min_evidence",
    )
    config["min_viability_score"] = cols[2].number_input(
        "Min Viability Score",
        min_value=0.0,
        max_value=100.0,
        value=_float_value(config.get("min_viability_score"), 70.0),
        step=1.0,
        format="%.1f",
        key="bankability_min_viability",
    )


def render_evidence_register_section(payload: dict) -> None:
    from ... import app as legacy

    st = legacy.st
    rows = payload.get("assumption_evidence", [])
    if not isinstance(rows, list):
        rows = []
    st.subheader("Evidence Register")
    edited = legacy._render_selectable_data_editor(
        rows,
        key="assumption_evidence_editor",
        label_builder=lambda row, index: legacy._editor_row_label(
            row,
            index,
            name_fields=("assumption", "value_reference", "category"),
            year_fields=("benchmark_year",),
            fallback_prefix="Evidence",
        ),
        num_rows="dynamic",
    )
    normalised = []
    for row in _editor_rows(edited):
        normalised.append(
            {
                "assumption": str(row.get("assumption", "") or "").strip(),
                "category": str(row.get("category", "General") or "General").strip(),
                "value_reference": str(row.get("value_reference", "") or "").strip(),
                "source": str(row.get("source", "") or "").strip(),
                "owner": str(row.get("owner", "") or "").strip(),
                "benchmark_year": str(row.get("benchmark_year", "") or "").strip(),
                "rationale": str(row.get("rationale", "") or "").strip(),
                "required": bool(row.get("required", True)),
            }
        )
    payload["assumption_evidence"] = normalised


def render_downside_case_section(payload: dict) -> None:
    from ... import app as legacy

    st = legacy.st
    rows = payload.get("downside_cases", [])
    if not isinstance(rows, list):
        rows = []
    st.subheader("Pharma Downside Cases")
    edited = legacy._render_selectable_data_editor(
        rows,
        key="downside_case_editor",
        label_builder=lambda row, index: legacy._editor_row_label(
            row,
            index,
            name_fields=("name",),
            fallback_prefix="Downside case",
        ),
        num_rows="dynamic",
    )
    normalised = []
    for row in _editor_rows(edited):
        normalised.append(
            {
                "name": str(row.get("name", "") or "").strip(),
                "approval_delay_years": max(_int_value(row.get("approval_delay_years"), 0), 0),
                "volume_multiplier": max(_float_value(row.get("volume_multiplier"), 1.0), 0.0),
                "price_multiplier": max(_float_value(row.get("price_multiplier"), 1.0), 0.0),
                "raw_material_multiplier": max(_float_value(row.get("raw_material_multiplier"), 1.0), 0.0),
                "direct_labor_multiplier": max(_float_value(row.get("direct_labor_multiplier"), 1.0), 0.0),
                "overhead_multiplier": max(_float_value(row.get("overhead_multiplier"), 1.0), 0.0),
                "receivable_days_delta": _int_value(row.get("receivable_days_delta"), 0),
                "inventory_days_delta": _int_value(row.get("inventory_days_delta"), 0),
                "payable_days_delta": _int_value(row.get("payable_days_delta"), 0),
                "capex_multiplier": max(_float_value(row.get("capex_multiplier"), 1.0), 0.0),
                "include_in_monte_carlo": bool(row.get("include_in_monte_carlo", True)),
            }
        )
    payload["downside_cases"] = normalised
