"""Generate human-readable insights from the analytical outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .config import FinanceAIConfig, get_default_config
from .forecasting import ForecastResult
try:
    from .data_quality import DataQualityResult
except ImportError:  # pragma: no cover - fallback for standalone execution
    from data_quality import DataQualityResult  # type: ignore


@dataclass(slots=True)
class InsightReport:
    headline: str
    highlights: List[str]
    category_breakdown: pd.DataFrame
    recurring_merchants: pd.DataFrame
    cashflow_metrics: Dict[str, float]
    forecast: ForecastResult
    quality_summary: pd.DataFrame
    anomaly_table: pd.DataFrame


def _build_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df.query("transaction_type == 'expense'")
    breakdown = (
        expenses.groupby("category")["amount"].sum().abs().sort_values(ascending=False)
    )
    breakdown = breakdown.reset_index().rename(columns={"amount": "total"})
    return breakdown


def _find_recurring_merchants(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df.query("transaction_type == 'expense'")
    grouped = expenses.groupby("merchant_clean").agg(
        total_amount=("amount", lambda x: -x.sum()),
        transactions=("amount", "count"),
        last_payment=("date", "max"),
    )
    recurring = grouped.query("transactions >= 3").sort_values("total_amount", ascending=False)
    return recurring.reset_index()


def _cashflow_metrics(df: pd.DataFrame, cfg: FinanceAIConfig) -> Dict[str, float]:
    total_expense = -df.query("transaction_type == 'expense' ")[cfg.amount_column].sum()
    total_income = df.query("transaction_type != 'expense'")[cfg.amount_column].sum()
    net_cashflow = total_income - total_expense
    average_daily_spend = (
        -df.query("transaction_type == 'expense'")
        .groupby("date_only")[cfg.amount_column]
        .sum()
        .mean()
    )
    return {
        "total_expense": float(total_expense),
        "total_income": float(total_income),
        "net_cashflow": float(net_cashflow),
        "average_daily_spend": float(average_daily_spend or 0.0),
    }


def _build_highlights(
    df: pd.DataFrame,
    breakdown: pd.DataFrame,
    metrics: Dict[str, float],
    forecast: ForecastResult,
    quality: Optional[DataQualityResult],
) -> List[str]:
    highlights: List[str] = []
    if not breakdown.empty:
        top_category = breakdown.iloc[0]
        highlights.append(
            f"Maior categoria de gasto: {top_category['category']} com R$ {top_category['total']:.2f}."
        )
    highlights.append(
        f"Ticket medio diario: R$ {metrics['average_daily_spend']:.2f}."
    )
    if not forecast.forecast.empty:
        next_month = forecast.forecast.iloc[0]
        highlights.append(
            f"Gasto previsto para o proximo mes: R$ {next_month:.2f}."
        )
    anomalies = df.query("is_anomaly == True")
    if not anomalies.empty:
        highlight_amt = -anomalies.groupby("merchant_clean")["amount"].sum().abs().max()
        highlights.append(
            f"Foram encontrados {len(anomalies)} gastos atipicos (valor maximo aproximado R$ {highlight_amt:.2f})."
        )
    if quality is not None and not quality.monthly_summary.empty:
        flagged = quality.monthly_summary.query("quality_flag == True")
        if not flagged.empty:
            month_col = flagged["month"]
            if pd.api.types.is_datetime64_any_dtype(month_col):
                month_labels = month_col.dt.strftime("%Y-%m").tolist()
            else:
                month_labels = month_col.astype(str).tolist()
            highlights.append(
                f"Meses com possivel inconsistencia de dados: {', '.join(month_labels)}."
            )
    return highlights


def generate_insight_report(
    dataframe: pd.DataFrame,
    forecast: ForecastResult,
    *,
    quality: Optional[DataQualityResult] = None,
    config: FinanceAIConfig | None = None,
) -> InsightReport:
    cfg = config or get_default_config()
    breakdown = _build_category_breakdown(dataframe)
    recurring = _find_recurring_merchants(dataframe)
    metrics = _cashflow_metrics(dataframe, cfg)
    highlights = _build_highlights(dataframe, breakdown, metrics, forecast, quality)
    anomaly_table = dataframe.query("is_anomaly == True").copy()
    anomaly_table = anomaly_table.sort_values("anomaly_score")
    quality_summary = quality.monthly_summary if quality is not None else pd.DataFrame()
    headline = "Panorama financeiro consolidado"
    return InsightReport(
        headline=headline,
        highlights=highlights,
        category_breakdown=breakdown,
        recurring_merchants=recurring,
        cashflow_metrics=metrics,
        forecast=forecast,
        quality_summary=quality_summary,
        anomaly_table=anomaly_table,
    )