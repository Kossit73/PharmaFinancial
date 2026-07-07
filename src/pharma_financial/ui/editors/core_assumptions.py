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
    years = payload.get("years", [])
    inflation_factors = legacy._inflation_factors_from_payload(payload)
    risk_factors = legacy._risk_factors_from_payload(payload)
    inflation_factor = inflation_factors[0] if inflation_factors else 1.0
    risk_factor = risk_factors[0] if risk_factors else 1.0

    editor_rows: list[dict] = []
    for row_product in rows:
        product_name = str(row_product.get(legacy.CORE_PRODUCT_FIELD, "") or "").strip()
        default_units = legacy._core_row_number(
            row_product,
            legacy.CORE_TOTAL_UNITS_FIELD,
            legacy._LEGACY_CORE_TOTAL_UNITS_FIELD,
        )
        default_capacity = legacy._core_row_number(
            row_product,
            legacy.CORE_CAPACITY_FIELD,
            legacy._LEGACY_CORE_CAPACITY_FIELD,
        )
        if default_capacity == 0.0 and product_name in capacity_defaults:
            default_capacity = float(capacity_defaults[product_name] or 0.0)
        if default_units == 0.0:
            default_units = legacy._resolved_total_production_units(
                product_name,
                float(total_unit_defaults.get(product_name, 0.0) or 0.0),
                default_capacity,
                years,
                production_estimate if isinstance(production_estimate, legacy.Mapping) else {},
            )
        production = legacy._core_row_number(row_product, legacy.CORE_PRODUCTION_COST_FIELD)
        selling = legacy._core_row_number(row_product, legacy.CORE_SELLING_PRICE_FIELD)
        freight = legacy._core_row_number(row_product, legacy.CORE_FREIGHT_COST_FIELD)
        markup = legacy._core_row_number(row_product, legacy.CORE_MARKUP_FIELD)
        editor_rows.append(
            legacy._build_core_assumption_row(
                product_name=product_name,
                production_cost=production,
                selling_price=selling,
                freight_cost=freight,
                markup_value=markup,
                total_units=default_units,
                max_capacity=default_capacity,
                years=years,
                production_estimate=production_estimate
                if isinstance(production_estimate, legacy.Mapping)
                else {},
                inflation_factor=inflation_factor,
                risk_factor=risk_factor,
            )
        )

    if not editor_rows:
        st.info("No core assumptions configured. Use the editor below or the add form to add entries.")

    st.caption(
        "Planned Total Units rescales the saved production curve across the full projection horizon. "
        "Year 1 Units, Year 1 Revenue, and Year 1 Cost are derived automatically from that curve."
    )

    edited_rows = legacy._render_selectable_data_editor(
        editor_rows,
        key="core_assumptions_editor",
        label_builder=lambda row, index: legacy._editor_row_label(
            row,
            index,
            name_fields=(legacy.CORE_PRODUCT_FIELD,),
            fallback_prefix="Product",
        ),
        column_config={
            legacy.CORE_PRODUCT_FIELD: st.column_config.TextColumn(
                "Product",
                required=True,
                help="Name of the product or production line.",
            ),
            legacy.CORE_PRODUCTION_COST_FIELD: st.column_config.NumberColumn(
                "Production Cost / Unit",
                min_value=0.0,
                format="%.4f",
            ),
            legacy.CORE_SELLING_PRICE_FIELD: st.column_config.NumberColumn(
                "Selling Price / Unit",
                min_value=0.0,
                format="%.4f",
            ),
            legacy.CORE_FREIGHT_COST_FIELD: st.column_config.NumberColumn(
                "Freight Cost / Unit",
                min_value=0.0,
                format="%.4f",
            ),
            legacy.CORE_MARKUP_FIELD: st.column_config.NumberColumn(
                "Markup / Unit",
                min_value=0.0,
                format="%.4f",
            ),
            legacy.CORE_TOTAL_UNITS_FIELD: st.column_config.NumberColumn(
                "Planned Total Units",
                min_value=0.0,
                step=1.0,
                format="%.4f",
                help=(
                    "Total units across the full projection horizon. Saving this value rescales the "
                    "existing year-by-year production profile instead of flattening it."
                ),
            ),
            legacy.CORE_CAPACITY_FIELD: st.column_config.NumberColumn(
                "Capacity Limit",
                min_value=0.0,
                step=1.0,
                format="%.4f",
                help=(
                    "Optional cap on Planned Total Units. Use the same full-projection unit basis as "
                    "Planned Total Units."
                ),
            ),
            legacy.CORE_YEAR1_UNITS_FIELD: st.column_config.NumberColumn(
                "Year 1 Units",
                disabled=True,
                format="%.4f",
            ),
            legacy.CORE_YEAR1_REVENUE_FIELD: st.column_config.NumberColumn(
                "Year 1 Revenue",
                disabled=True,
                format="%.4f",
            ),
            legacy.CORE_YEAR1_COST_FIELD: st.column_config.NumberColumn(
                "Year 1 Cost",
                disabled=True,
                format="%.4f",
            ),
        },
        column_order=[
            legacy.CORE_PRODUCT_FIELD,
            legacy.CORE_PRODUCTION_COST_FIELD,
            legacy.CORE_SELLING_PRICE_FIELD,
            legacy.CORE_FREIGHT_COST_FIELD,
            legacy.CORE_MARKUP_FIELD,
            legacy.CORE_TOTAL_UNITS_FIELD,
            legacy.CORE_CAPACITY_FIELD,
            legacy.CORE_YEAR1_UNITS_FIELD,
            legacy.CORE_YEAR1_REVENUE_FIELD,
            legacy.CORE_YEAR1_COST_FIELD,
        ],
        num_rows="dynamic",
        row_caption=(
            "Edit one product below, then click Save row. Planned Total Units rescales the saved "
            "production curve and refreshes the derived Year 1 outputs."
        ),
        full_caption=(
            "Edit the full table below, then click Apply table changes. Derived Year 1 outputs "
            "refresh after the draft is applied."
        ),
    )

    updated_rows: list[dict] = []
    capped_products: list[str] = []
    for row in edited_rows:
        product = str(row.get(legacy.CORE_PRODUCT_FIELD, "") or "").strip()
        if not product:
            continue
        production = legacy._core_row_number(row, legacy.CORE_PRODUCTION_COST_FIELD)
        selling = legacy._core_row_number(row, legacy.CORE_SELLING_PRICE_FIELD)
        freight = legacy._core_row_number(row, legacy.CORE_FREIGHT_COST_FIELD)
        markup = legacy._core_row_number(row, legacy.CORE_MARKUP_FIELD)
        requested_units = max(
            legacy._core_row_number(row, legacy.CORE_TOTAL_UNITS_FIELD),
            0.0,
        )
        max_capacity = max(
            legacy._core_row_number(row, legacy.CORE_CAPACITY_FIELD),
            0.0,
        )

        updated_row = legacy._build_core_assumption_row(
            product_name=product,
            production_cost=production,
            selling_price=selling,
            freight_cost=freight,
            markup_value=markup,
            total_units=requested_units,
            max_capacity=max_capacity,
            years=years,
            production_estimate=production_estimate
            if isinstance(production_estimate, legacy.Mapping)
            else {},
            inflation_factor=inflation_factor,
            risk_factor=risk_factor,
        )
        if (
            max_capacity > 0.0
            and float(updated_row[legacy.CORE_TOTAL_UNITS_FIELD]) < requested_units - 1e-9
        ):
            capped_products.append(product)
        updated_rows.append(updated_row)

    if capped_products:
        st.warning(
            "Planned total units were capped at the Capacity Limit for: "
            + ", ".join(capped_products)
            + "."
        )

    st.session_state["core_assumption_rows"] = updated_rows
    legacy._prime_core_widget_state(updated_rows)

    st.markdown("#### Add a core assumption")
    with st.form("add_core_assumption"):
        new_product = st.text_input(
            "Product", key="core_new_description", help="Name of the product or production line."
        )
        new_production = st.number_input(
            "Production Cost / Unit", value=0.0, step=0.001, format="%.4f", key="core_new_prod"
        )
        new_selling = st.number_input(
            "Selling Price / Unit", value=0.0, step=0.001, format="%.4f", key="core_new_sell"
        )
        new_freight = st.number_input(
            "Freight Cost / Unit", value=0.0, step=0.001, format="%.4f", key="core_new_freight"
        )
        new_markup = st.number_input(
            "Markup / Unit", value=0.0, step=0.01, format="%.2f", key="core_new_markup"
        )
        new_units = st.number_input(
            "Planned Total Units",
            value=0.0,
            step=1.0,
            format="%.4f",
            key="core_new_units",
            min_value=0.0,
        )
        new_capacity = st.number_input(
            "Capacity Limit",
            value=0.0,
            step=1.0,
            format="%.4f",
            key="core_new_capacity",
            min_value=0.0,
        )
        submitted = st.form_submit_button("Add")

    if submitted:
        if not new_product.strip():
            st.warning("Product is required to add a core assumption.")
        else:
            new_row = legacy._build_core_assumption_row(
                product_name=new_product.strip(),
                production_cost=float(new_production),
                selling_price=float(new_selling),
                freight_cost=float(new_freight),
                markup_value=float(new_markup),
                total_units=float(new_units),
                max_capacity=float(new_capacity),
                years=years,
                production_estimate=production_estimate
                if isinstance(production_estimate, legacy.Mapping)
                else {},
                inflation_factor=inflation_factor,
                risk_factor=risk_factor,
            )
            if (
                new_capacity > 0.0
                and float(new_row[legacy.CORE_TOTAL_UNITS_FIELD]) < float(new_units) - 1e-9
            ):
                st.warning("Planned total units were capped at the Capacity Limit.")
            rows.append(new_row)
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
