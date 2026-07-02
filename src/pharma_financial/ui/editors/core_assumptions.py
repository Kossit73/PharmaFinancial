"""Core assumption editor extracted from the legacy input landing page."""

from __future__ import annotations


def render_core_assumptions_section(payload: dict) -> None:
    from ... import app as legacy

    st = legacy.st
    st.subheader("Core Assumptions")
    rows = st.session_state.get("core_assumption_rows", [])
    if not rows:
        rows = legacy._payload_to_core_rows(payload)
        st.session_state["core_assumption_rows"] = rows
    legacy._prime_core_widget_state(rows)

    production_estimate = payload.get("production_estimate", {})
    total_unit_defaults = payload.get("total_production_units", {})
    capacity_defaults = payload.get("production_capacity", {})

    editor_rows: list[dict] = []
    for row_product in rows:
        product_name = str(row_product.get("Product", "") or "").strip()
        default_units = float(row_product.get("Total Production Units", 0.0) or 0.0)
        if default_units == 0.0:
            if product_name in total_unit_defaults:
                default_units = float(total_unit_defaults[product_name] or 0.0)
            elif (
                isinstance(production_estimate, legacy.Mapping)
                and product_name in production_estimate
            ):
                existing_series = production_estimate.get(product_name, [])
                if isinstance(existing_series, legacy.Sequence) and existing_series:
                    default_units = float(existing_series[0] or 0.0)
        default_capacity = float(row_product.get("Max Capacity", 0.0) or 0.0)
        if default_capacity == 0.0 and product_name in capacity_defaults:
            default_capacity = float(capacity_defaults[product_name] or 0.0)
        clamped_units = (
            min(default_units, default_capacity)
            if default_capacity > 0.0
            else default_units
        )
        production = float(row_product.get("Production Cost", 0.0) or 0.0)
        selling = float(row_product.get("Selling Price", 0.0) or 0.0)
        freight = float(row_product.get("Freight Cost", 0.0) or 0.0)
        markup = float(row_product.get("Markup", 0.0) or 0.0)
        editor_rows.append(
            {
                "Product": product_name,
                "Production Cost": production,
                "Selling Price": selling,
                "Freight Cost": freight,
                "Markup": markup,
                "Total Production Units": clamped_units,
                "Max Capacity": default_capacity,
                "Total Revenue": float(clamped_units) * selling,
                "Total Cost": float(clamped_units) * (production + freight + markup),
            }
        )

    if not editor_rows:
        st.info("No core assumptions configured. Use the editor below or the add form to add entries.")

    edited_rows = legacy._render_selectable_data_editor(
        editor_rows,
        key="core_assumptions_editor",
        label_builder=lambda row, index: legacy._editor_row_label(
            row,
            index,
            name_fields=("Product",),
            fallback_prefix="Product",
        ),
        column_config={
            "Total Revenue": st.column_config.NumberColumn(
                "Total Revenue",
                disabled=True,
                format="%.4f",
            ),
            "Total Cost": st.column_config.NumberColumn(
                "Total Cost",
                disabled=True,
                format="%.4f",
            ),
        },
        column_order=[
            "Product",
            "Production Cost",
            "Selling Price",
            "Freight Cost",
            "Markup",
            "Total Production Units",
            "Max Capacity",
            "Total Revenue",
            "Total Cost",
        ],
        num_rows="dynamic",
    )

    updated_rows: list[dict] = []
    capped_products: list[str] = []
    for index, row in enumerate(edited_rows):
        product = str(row.get("Product", "") or "").strip()
        if not product:
            continue
        production = float(row.get("Production Cost", 0.0) or 0.0)
        selling = float(row.get("Selling Price", 0.0) or 0.0)
        freight = float(row.get("Freight Cost", 0.0) or 0.0)
        markup = float(row.get("Markup", 0.0) or 0.0)
        total_units = max(float(row.get("Total Production Units", 0.0) or 0.0), 0.0)
        max_capacity = max(float(row.get("Max Capacity", 0.0) or 0.0), 0.0)
        clamped_units = total_units
        if max_capacity > 0.0 and total_units > max_capacity + 1e-9:
            clamped_units = max_capacity
            capped_products.append(product)

        legacy._scaled_production_series(
            product,
            clamped_units,
            payload.get("years", []),
            production_estimate,
        )
        total_revenue = clamped_units * selling
        total_cost = clamped_units * (production + freight + markup)
        updated_rows.append(
            {
                "Product": product,
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

    if capped_products:
        st.warning(
            "Total production units were capped at max capacity for: "
            + ", ".join(capped_products)
            + "."
        )

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
