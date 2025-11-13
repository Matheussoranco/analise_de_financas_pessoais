"""Visualization utilities powered by Plotly."""

from __future__ import annotations

import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore

try:
    from .forecasting import ForecastResult
except ImportError:  # executed as top-level script
    from forecasting import ForecastResult  # type: ignore


def spending_over_time(dataframe: pd.DataFrame) -> go.Figure:
    daily = (
        dataframe
        .query("transaction_type == 'expense'")
        .groupby("date_only")["amount"]
        .sum()
        .mul(-1)
        .reset_index(name="daily_spend")
    )
    fig = px.line(
        daily,
        x="date_only",
        y="daily_spend",
        title="Gastos diários",
        labels={"date_only": "Data", "daily_spend": "Gasto (R$)"},
    )
    return fig


def category_breakdown_chart(breakdown: pd.DataFrame) -> go.Figure:
    fig = px.treemap(
        breakdown,
        path=["category"],
        values="total",
        title="Distribuição de gastos por categoria",
    )
    return fig


def forecast_chart(result: ForecastResult) -> go.Figure:
    fig = go.Figure()
    if not result.history.empty:
        fig.add_trace(
            go.Scatter(
                x=result.history.index,
                y=result.history.values,
                name="Histórico",
                mode="lines+markers",
            )
        )
    if not result.forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=result.forecast.index,
                y=result.forecast.values,
                name="Previsão",
                mode="lines+markers",
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(
        title="Projeção mensal de gastos",
        xaxis_title="Mês",
        yaxis_title="Gasto (R$)",
    )
    return fig


def anomaly_scatter(dataframe: pd.DataFrame) -> go.Figure:
    anomalies = dataframe.query("is_anomaly == True")
    if anomalies.empty:
        fig = go.Figure()
        fig.update_layout(title="Nenhum gasto atípico identificado")
        return fig
    fig = px.scatter(
        anomalies,
        x="date",
        y="amount",
        color="category",
        hover_data=["merchant_clean", "anomaly_score", "source_file"],
        title="Transações atípicas",
    )
    return fig