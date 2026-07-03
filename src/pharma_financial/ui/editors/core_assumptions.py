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

    years = payload.get("years", [])
    editor_rows = [dict(row) for row in rows]

    if not editor_rows:
        st.info("No core assumptions configured. Use the editor below or the add form to add entries.")

    st.caption(
        "Yearly Total Units Produced is the Year 1 production quantity. Use the yearly schedule below to "
        "calculate later years with the Yearly Increment Tool or manual edits."
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
                "Yearly Total Units Produced",
                min_value=0.0,
                step=1.0,
                format="%.4f",
                help=(
                    "Year 1 production quantity. Later years stay editable in the yearly schedule below."
                ),
            ),
            legacy.CORE_CAPACITY_FIELD: st.column_config.NumberColumn(
                "Capacity Limit",
                min_value=0.0,
                step=1.0,
                format="%.4f",
                help=(
                    "Optional Year 1 capacity reference. Use the yearly schedule below if capacity changes by year."
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
            "Edit one product below, then click Save row. These values define the editable Year 1 baseline."
        ),
        full_caption=(
            "Edit the full table below, then click Apply table changes. These values remain editable and are not auto-rescaled."
        ),
    )

    updated_rows: list[dict] = []
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
            total_units=max(requested_units, 0.0),
            max_capacity=max(max_capacity, 0.0),
            years=years,
            production_estimate={},
        )
        updated_rows.append(updated_row)

    st.session_state["core_assumption_rows"] = updated_rows
    legacy._prime_core_widget_state(updated_rows)

    schedule_editor_key = "core_assumptions_schedule_editor"
    schedule_rows = legacy._core_schedule_rows_from_rows(updated_rows, payload)
    saved_schedule_rows, draft_schedule_rows = legacy._initialise_selectable_editor_state(
        schedule_editor_key,
        schedule_rows,
    )
    working_schedule_rows = (
        draft_schedule_rows
        if not legacy._editor_rows_equal(saved_schedule_rows, draft_schedule_rows)
        else saved_schedule_rows
    )

    st.markdown("#### Yearly Increment Tool")
    st.caption(
        "Use the saved Year 1 baseline above, then populate each later year below with an increment or direct edits."
    )
    yearly_increment = st.number_input(
        "Yearly Increment %",
        value=float(st.session_state.get("core_schedule_increment_pct", 0.0) or 0.0),
        step=0.1,
        key="core_schedule_increment_pct",
    )
    target_column = st.selectbox(
        "Apply increment to",
        [
            "All Supported Fields",
            "Production Cost / Unit",
            "Selling Price / Unit",
            "Freight Cost / Unit",
            "Markup / Unit",
            "Yearly Total Units Produced",
            "Capacity Limit",
        ],
        key="core_schedule_increment_target",
    )
    increment_cols = st.columns(3)
    preview_increment = increment_cols[0].button(
        "Preview Yearly Increment",
        key="core_schedule_increment_preview",
    )
    apply_increment = increment_cols[1].button(
        "Apply Yearly Increment",
        key="core_schedule_increment_apply",
    )
    cancel_increment = increment_cols[2].button(
        "Cancel Yearly Increment",
        key="core_schedule_increment_cancel",
    )

    if preview_increment or apply_increment:
        selected_fields = (
            (
                legacy.CORE_PRODUCTION_COST_FIELD,
                legacy.CORE_SELLING_PRICE_FIELD,
                legacy.CORE_FREIGHT_COST_FIELD,
                legacy.CORE_MARKUP_FIELD,
                legacy.CORE_TOTAL_UNITS_FIELD,
                legacy.CORE_CAPACITY_FIELD,
            )
            if target_column == "All Supported Fields"
            else (
                legacy.CORE_PRODUCTION_COST_FIELD,
            )
            if target_column == "Production Cost / Unit"
            else (
                legacy.CORE_SELLING_PRICE_FIELD,
            )
            if target_column == "Selling Price / Unit"
            else (
                legacy.CORE_FREIGHT_COST_FIELD,
            )
            if target_column == "Freight Cost / Unit"
            else (
                legacy.CORE_MARKUP_FIELD,
            )
            if target_column == "Markup / Unit"
            else (
                legacy.CORE_TOTAL_UNITS_FIELD,
            )
            if target_column == "Yearly Total Units Produced"
            else (legacy.CORE_CAPACITY_FIELD,)
        )
        incremented_rows = legacy._apply_grouped_yearly_increment(
            working_schedule_rows,
            group_field=legacy.CORE_PRODUCT_FIELD,
            target_fields=selected_fields,
            increment_pct=float(yearly_increment),
            integer_fields=(
                legacy.CORE_TOTAL_UNITS_FIELD,
                legacy.CORE_CAPACITY_FIELD,
            ),
        )
        if preview_increment:
            legacy._set_editor_draft_rows(
                schedule_editor_key,
                incremented_rows,
                refresh_widgets=True,
            )
        else:
            legacy._commit_editor_rows(
                schedule_editor_key,
                incremented_rows,
                message="Applied yearly increment.",
            )
        legacy._rerun()

    if cancel_increment and not legacy._editor_rows_equal(saved_schedule_rows, draft_schedule_rows):
        legacy._discard_editor_draft(
            schedule_editor_key,
            message="Discarded yearly increment draft.",
        )
        legacy._rerun()

    st.markdown("#### Core Assumptions by Year")
    schedule_rows = legacy._render_selectable_data_editor(
        schedule_rows,
        key=schedule_editor_key,
        label_builder=lambda row, index: legacy._editor_row_label(
            row,
            index,
            name_fields=(legacy.CORE_PRODUCT_FIELD,),
            year_fields=(legacy.CORE_SCHEDULE_YEAR_FIELD,),
            fallback_prefix="Core row",
        ),
        column_order=[
            legacy.CORE_SCHEDULE_YEAR_FIELD,
            legacy.CORE_PRODUCT_FIELD,
            legacy.CORE_PRODUCTION_COST_FIELD,
            legacy.CORE_SELLING_PRICE_FIELD,
            legacy.CORE_FREIGHT_COST_FIELD,
            legacy.CORE_MARKUP_FIELD,
            legacy.CORE_TOTAL_UNITS_FIELD,
            legacy.CORE_CAPACITY_FIELD,
        ],
        column_config={
            legacy.CORE_SCHEDULE_YEAR_FIELD: st.column_config.NumberColumn(
                "Year",
                disabled=True,
                format="%d",
            ),
            legacy.CORE_PRODUCT_FIELD: st.column_config.TextColumn(
                "Product",
                disabled=True,
            ),
            legacy.CORE_PRODUCTION_COST_FIELD: st.column_config.NumberColumn(
                "Production Cost / Unit",
                min_value=0.0,
                step=0.001,
                format="%.4f",
            ),
            legacy.CORE_SELLING_PRICE_FIELD: st.column_config.NumberColumn(
                "Selling Price / Unit",
                min_value=0.0,
                step=0.001,
                format="%.4f",
            ),
            legacy.CORE_FREIGHT_COST_FIELD: st.column_config.NumberColumn(
                "Freight Cost / Unit",
                min_value=0.0,
                step=0.001,
                format="%.4f",
            ),
            legacy.CORE_MARKUP_FIELD: st.column_config.NumberColumn(
                "Markup / Unit",
                min_value=0.0,
                step=0.001,
                format="%.4f",
            ),
            legacy.CORE_TOTAL_UNITS_FIELD: st.column_config.NumberColumn(
                "Yearly Total Units Produced",
                min_value=0.0,
                step=1.0,
                format="%.4f",
            ),
            legacy.CORE_CAPACITY_FIELD: st.column_config.NumberColumn(
                "Capacity Limit",
                min_value=0.0,
                step=1.0,
                format="%.4f",
            ),
        },
        num_rows="fixed",
        row_caption=(
            "Select one product-year row below, edit the values, then click Save row."
        ),
        full_caption=(
            "Edit the full yearly schedule below, then click Apply table changes."
        ),
    )
    legacy._core_schedule_rows_to_payload(schedule_rows, payload)
    st.session_state["core_assumption_rows"] = legacy._payload_to_core_rows(payload)
    legacy._prime_core_widget_state(st.session_state["core_assumption_rows"])

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
            "Yearly Total Units Produced",
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
                production_estimate={},
            )
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
