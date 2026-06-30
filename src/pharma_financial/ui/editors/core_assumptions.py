"""Core assumption editor extracted from the legacy input landing page."""

from __future__ import annotations


def render_core_assumptions_section(payload: dict) -> None:
    from ... import app as legacy

    st = legacy.st
    st.subheader("Core Assumptions")
    rows = st.session_state.get("core_assumption_rows", [])
    legacy._prime_core_widget_state(rows)

    if not rows:
        st.info("No core assumptions configured. Use the form below to add entries.")

    production_estimate = payload.get("production_estimate", {})
    total_unit_defaults = payload.get("total_production_units", {})
    capacity_defaults = payload.get("production_capacity", {})

    updated_rows: list[dict] = []
    for index, row in enumerate(rows):
        container = st.container()
        with container:
            row_product = str(row.get("Product", ""))
            default_units = float(row.get("Total Production Units", 0.0))
            if default_units == 0.0:
                if row_product in total_unit_defaults:
                    default_units = float(total_unit_defaults[row_product])
                elif isinstance(production_estimate, legacy.Mapping) and row_product in production_estimate:
                    existing_series = production_estimate.get(row_product, [])
                    if isinstance(existing_series, legacy.Sequence) and existing_series:
                        default_units = float(existing_series[0])
            default_capacity = float(row.get("Max Capacity", 0.0))
            if default_capacity == 0.0 and row_product in capacity_defaults:
                default_capacity = float(capacity_defaults[row_product])

            cols = st.columns([3, 2, 2, 2, 2, 2, 2, 2, 2, 1])
            desc_key = f"core_desc_{index}"
            prod_key = f"core_prod_{index}"
            sell_key = f"core_sell_{index}"
            freight_key = f"core_freight_{index}"
            markup_key = f"core_markup_{index}"
            units_key = f"core_units_{index}"
            capacity_key = f"core_capacity_{index}"

            legacy._ensure_widget_default(desc_key, row.get("Product", ""))
            legacy._ensure_widget_default(prod_key, float(row.get("Production Cost", 0.0)))
            legacy._ensure_widget_default(sell_key, float(row.get("Selling Price", 0.0)))
            legacy._ensure_widget_default(freight_key, float(row.get("Freight Cost", 0.0)))
            legacy._ensure_widget_default(markup_key, float(row.get("Markup", 0.0)))
            legacy._ensure_widget_default(units_key, default_units)
            legacy._ensure_widget_default(capacity_key, default_capacity)

            description = cols[0].text_input(
                "Description",
                value=str(st.session_state.get(desc_key, "")),
                key=desc_key,
                help="Name of the product or assumption this row represents.",
            )
            production = cols[1].number_input(
                "Production Cost",
                value=float(st.session_state.get(prod_key, 0.0)),
                key=prod_key,
                step=0.001,
                format="%.4f",
            )
            selling = cols[2].number_input(
                "Selling Price",
                value=float(st.session_state.get(sell_key, 0.0)),
                key=sell_key,
                step=0.001,
                format="%.4f",
            )
            freight = cols[3].number_input(
                "Freight Cost",
                value=float(st.session_state.get(freight_key, 0.0)),
                key=freight_key,
                step=0.001,
                format="%.4f",
            )
            markup = cols[4].number_input(
                "Markup",
                value=float(st.session_state.get(markup_key, 0.0)),
                key=markup_key,
                step=0.01,
                format="%.2f",
            )
            total_units = cols[5].number_input(
                "Total Production Units",
                value=float(st.session_state.get(units_key, default_units)),
                key=units_key,
                step=1.0,
                format="%.4f",
                min_value=0.0,
            )
            max_capacity = cols[6].number_input(
                "Max Capacity",
                value=float(st.session_state.get(capacity_key, default_capacity)),
                key=capacity_key,
                step=1.0,
                format="%.4f",
                min_value=0.0,
            )

            clamped_units = float(total_units)
            if max_capacity > 0.0 and clamped_units > max_capacity + 1e-9:
                cols[5].error("Capacity exceeded")
                clamped_units = max_capacity

            legacy._scaled_production_series(
                description.strip(),
                clamped_units,
                payload.get("years", []),
                production_estimate,
            )
            total_revenue = float(clamped_units) * float(selling)
            total_cost = float(clamped_units) * (
                float(production) + float(freight) + float(markup)
            )

            revenue_key = f"core_revenue_{index}"
            cost_key = f"core_cost_{index}"
            legacy._set_widget_value(revenue_key, total_revenue)
            legacy._set_widget_value(cost_key, total_cost)

            cols[7].number_input(
                "Total Revenue",
                key=revenue_key,
                format="%.4f",
                disabled=True,
            )
            cols[8].number_input(
                "Total Cost",
                key=cost_key,
                format="%.4f",
                disabled=True,
            )
            if cols[9].button("Remove", key=f"core_remove_{index}"):
                del rows[index]
                st.session_state["core_assumption_rows"] = rows
                legacy._prime_core_widget_state(rows)
                legacy._rerun()

        updated_rows.append(
            {
                "Product": description.strip(),
                "Production Cost": production,
                "Selling Price": selling,
                "Freight Cost": freight,
                "Markup": markup,
                "Total Production Units": clamped_units,
                "Max Capacity": max_capacity,
                "Total Revenue": total_revenue,
                "Total Cost": total_cost,
            }
        )

    if updated_rows != rows:
        st.session_state["core_assumption_rows"] = updated_rows
        legacy._prime_core_widget_state(updated_rows)

    st.markdown("#### Add a core assumption")
    with st.form("add_core_assumption"):
        new_description = st.text_input(
            "Description", key="core_new_description", help="Label for the new row."
        )
        new_production = st.number_input(
            "Production Cost", value=0.0, step=0.001, format="%.4f", key="core_new_prod"
        )
        new_selling = st.number_input(
            "Selling Price", value=0.0, step=0.001, format="%.4f", key="core_new_sell"
        )
        new_freight = st.number_input(
            "Freight Cost", value=0.0, step=0.001, format="%.4f", key="core_new_freight"
        )
        new_markup = st.number_input(
            "Markup", value=0.0, step=0.01, format="%.2f", key="core_new_markup"
        )
        new_units = st.number_input(
            "Total Production Units",
            value=0.0,
            step=1.0,
            format="%.4f",
            key="core_new_units",
            min_value=0.0,
        )
        new_capacity = st.number_input(
            "Max Capacity",
            value=0.0,
            step=1.0,
            format="%.4f",
            key="core_new_capacity",
            min_value=0.0,
        )
        submitted = st.form_submit_button("Add")

    if submitted:
        if not new_description.strip():
            st.warning("Description is required to add a core assumption.")
        else:
            clamped_units = new_units
            if new_capacity > 0.0 and new_units > new_capacity + 1e-9:
                st.error("Capacity exceeded")
                clamped_units = new_capacity
            total_revenue = clamped_units * new_selling
            total_cost = clamped_units * (new_production + new_freight + new_markup)
            rows.append(
                {
                    "Product": new_description.strip(),
                    "Production Cost": new_production,
                    "Selling Price": new_selling,
                    "Freight Cost": new_freight,
                    "Markup": new_markup,
                    "Total Production Units": clamped_units,
                    "Max Capacity": new_capacity,
                    "Total Revenue": total_revenue,
                    "Total Cost": total_cost,
                }
            )
            st.session_state["core_assumption_rows"] = rows
            legacy._prime_core_widget_state(rows)
            for key in (
                "core_new_description",
                "core_new_prod",
                "core_new_sell",
                "core_new_freight",
                "core_new_markup",
                "core_new_units",
                "core_new_capacity",
            ):
                st.session_state.pop(key, None)
            legacy._rerun()

