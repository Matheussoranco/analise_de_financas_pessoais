"""Feature engineering block for the finance AI pipeline."""

from __future__ import annotations

import pandas as pd

from .config import FinanceAIConfig, get_default_config


def engineer_features(
	dataframe: pd.DataFrame,
	*,
	config: FinanceAIConfig | None = None,
) -> pd.DataFrame:
	cfg = config or get_default_config()
	df = dataframe.copy()
	df = df.sort_values(cfg.date_column)
	df.set_index(cfg.date_column, inplace=True)

	expenses = df["amount"].where(df["transaction_type"] == "expense", 0.0)
	incomes = df["amount"].where(df["transaction_type"] != "expense", 0.0)

	df["rolling_7d_spend"] = expenses.rolling("7D", min_periods=1).sum()
	df["rolling_30d_spend"] = expenses.rolling("30D", min_periods=1).sum()
	df["rolling_90d_spend"] = expenses.rolling("90D", min_periods=1).sum()
	df["rolling_7d_income"] = incomes.rolling("7D", min_periods=1).sum()
	df["rolling_30d_income"] = incomes.rolling("30D", min_periods=1).sum()

	daily_expenses = expenses.resample("D").sum()
	daily_mean = daily_expenses.rolling(window=30, min_periods=7).mean()
	daily_std = daily_expenses.rolling(window=30, min_periods=7).std(ddof=0)
	zscore = (daily_expenses - daily_mean) / daily_std
	zscore = zscore.reindex(df.index, method="ffill")
	df["daily_spend_zscore"] = zscore.fillna(0.0)

	df["month_total_expense"] = expenses.resample("M").sum().reindex(df.index, method="ffill")
	df["month_total_income"] = incomes.resample("M").sum().reindex(df.index, method="ffill")

	df.reset_index(inplace=True)
	return df
