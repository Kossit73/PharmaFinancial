"""Dash web application exposing the Longevity Pharmaceuticals financial model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import dash
from dash import Dash, Input, Output, State, dash_table, dcc, html
import plotly.express as px

from .inputs import load_inputs
from .model import FinancialModel


def serve_layout() -> html.Div:
    inputs = load_inputs()
    model = FinancialModel(inputs)
    outputs = model.run()

    return html.Div(
        [
            html.H1("Longevity Pharmaceuticals Financial Model"),
            dcc.Store(id="model-store", data=serialize_outputs(outputs)),
            dcc.Tabs(
                [
                    dcc.Tab(label="Input Landing Page", children=input_layout(inputs)),
                    dcc.Tab(label="Financial Dashboard", children=dashboard_layout(outputs)),
                    dcc.Tab(label="Statements", children=statement_layout(outputs)),
                    dcc.Tab(label="Scenario & Sensitivity", children=analysis_layout(outputs)),
                    dcc.Tab(label="Monte Carlo Simulation", children=monte_carlo_layout(outputs)),
                ]
            ),
        ]
    )


def input_layout(inputs) -> html.Div:
    return html.Div(
        [
            html.H3("Model Assumptions"),
            html.P("Default assumptions sourced from the Longevity Pharmaceuticals project."),
            dash_table.DataTable(
                id="assumptions-table",
                data=[
                    {
                        "Product": name,
                        "Production Cost": params.production_cost,
                        "Selling Price": params.selling_price,
                        "Freight Cost": params.freight_cost,
                        "Markup": params.markup,
                    }
                    for name, params in inputs.unit_costs.items()
                ],
                columns=[
                    {"name": col, "id": col}
                    for col in ["Product", "Production Cost", "Selling Price", "Freight Cost", "Markup"]
                ],
                style_table={"overflowX": "auto"},
            ),
            html.Br(),
            html.P(
                "Use the CLI (python -m pharma_financial.cli) to update the JSON inputs "
                "and refresh the web dashboard with bespoke assumptions."
            ),
        ]
    )


def dashboard_layout(outputs) -> html.Div:
    income = outputs.income_statement.reset_index().rename(columns={"index": "Year"})
    fig_revenue = px.line(income, x="Year", y="Net Revenue", title="Net Revenue")
    fig_ebitda = px.line(income, x="Year", y="EBITDA", title="EBITDA")

    summary = outputs.summary_metrics.reset_index().rename(columns={"index": "Metric"})

    return html.Div(
        [
            html.H3("Key Financial Metrics"),
            dcc.Graph(figure=fig_revenue),
            dcc.Graph(figure=fig_ebitda),
            dash_table.DataTable(
                data=summary.to_dict("records"),
                columns=[{"name": c, "id": c} for c in summary.columns],
                style_table={"overflowX": "auto"},
            ),
        ]
    )


def statement_layout(outputs) -> html.Div:
    return html.Div(
        [
            html.H3("Statements"),
            html.H4("Statement of Financial Performance"),
            data_table(outputs.income_statement),
            html.H4("Statement of Financial Position"),
            data_table(outputs.balance_sheet),
            html.H4("Statement of Cash Flows"),
            data_table(outputs.cash_flow),
        ]
    )


def analysis_layout(outputs) -> html.Div:
    scenario_tables = [
        html.Div(
            [
                html.H4(f"Scenario: {name}"),
                data_table(df),
            ]
        )
        for name, df in outputs.scenario_results.items()
    ]
    sensitivity_tables = [
        html.Div(
            [
                html.H4(f"Sensitivity: {variable}"),
                data_table(df),
            ]
        )
        for variable, df in outputs.sensitivity_results.items()
    ]
    return html.Div(scenario_tables + sensitivity_tables)


def monte_carlo_layout(outputs) -> html.Div:
    fig = px.histogram(outputs.monte_carlo, x="NPV", nbins=30, title="Monte Carlo NPV Distribution")
    return html.Div(
        [
            html.H3("Monte Carlo Simulation"),
            dcc.Graph(figure=fig),
        ]
    )


def data_table(df) -> dash_table.DataTable:
    table = df.reset_index()
    table = table.rename(columns={"index": "Year"})
    return dash_table.DataTable(
        data=table.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table.columns],
        style_table={"overflowX": "auto"},
    )


def serialize_outputs(outputs) -> Dict[str, dict]:
    return {
        "income_statement": outputs.income_statement.to_dict(),
        "balance_sheet": outputs.balance_sheet.to_dict(),
        "cash_flow": outputs.cash_flow.to_dict(),
    }


def create_app() -> Dash:
    app = dash.Dash(__name__)
    app.layout = serve_layout
    return app


def main() -> None:
    app = create_app()
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
