"""Expense forecasting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

from .config import FinanceAIConfig, get_default_config


@dataclass(slots=True)
class ForecastResult:
    history: pd.Series
    forecast: pd.Series
    model_summary: str


def _prepare_monthly_expenses(
    dataframe: pd.DataFrame,
    config: FinanceAIConfig,
) -> pd.Series:
    expenses = dataframe.query("transaction_type == 'expense'")
    monthly = expenses.groupby("month")[config.amount_column].sum()
    monthly = monthly.fillna(0.0)
    monthly = monthly.sort_index()
    monthly = -monthly  # convert expenses to positive magnitudes
    monthly.name = "monthly_expense"
    return monthly


def forecast_expenses(
    dataframe: pd.DataFrame,
    *,
    config: Optional[FinanceAIConfig] = None,
) -> ForecastResult:
    cfg = config or get_default_config()
    monthly_series = _prepare_monthly_expenses(dataframe, cfg)
    if len(monthly_series) < 3:
        mean_value = monthly_series.mean() if len(monthly_series) else 0.0
        future_index = pd.date_range(
            start=monthly_series.index[-1] if len(monthly_series) else pd.Timestamp.utcnow(),
            periods=max(cfg.forecast.horizon_months, 1) + 1,
            freq="M",
        )[1:]
        forecast = pd.Series(mean_value, index=future_index, name="forecast")
        return ForecastResult(history=monthly_series, forecast=forecast, model_summary="Media simples por falta de dados.")

    try:
        model = ExponentialSmoothing(
            monthly_series,
            trend="add",
            seasonal="mul",
            seasonal_periods=cfg.forecast.seasonal_periods,
            damped_trend=cfg.forecast.damped_trend,
        )
        fitted = model.fit(optimized=True, use_brute=True)
        future = fitted.forecast(cfg.forecast.horizon_months)
        summary = fitted.summary().as_text() if hasattr(fitted, "summary") else "Modelo Holt-Winters ajustado."
        future.name = "forecast"
        return ForecastResult(history=monthly_series, forecast=future, model_summary=summary)
    except Exception as exc:  # best-effort fallback
        mean_value = monthly_series.mean()
        future_index = pd.date_range(
            start=monthly_series.index[-1],
            periods=cfg.forecast.horizon_months + 1,
            freq="M",
        )[1:]
        forecast = pd.Series(mean_value, index=future_index, name="forecast")
        return ForecastResult(
            history=monthly_series,
            forecast=forecast,
            model_summary=f"Holt-Winters falhou ({exc}); utilizando media.",
        )