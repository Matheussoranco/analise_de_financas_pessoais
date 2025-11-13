"""Preprocessing routines for credit card statements."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import FinanceAIConfig, get_default_config


def _classify_transaction(
	description: str,
	amount: float,
	config: FinanceAIConfig,
) -> Tuple[str, float]:
	abs_amount = abs(amount)
	if config.is_income(description, amount):
		return "income", abs_amount
	if config.is_refund(description, amount):
		return "refund", abs_amount
	return "expense", -abs_amount


def prepare_transactions(
	dataframe: pd.DataFrame,
	*,
	config: FinanceAIConfig | None = None,
) -> pd.DataFrame:
	cfg = config or get_default_config()
	df = dataframe.copy()
	df[cfg.date_column] = pd.to_datetime(df[cfg.date_column], errors="coerce")
	df = df.dropna(subset=[cfg.date_column])
	df[cfg.description_column] = df[cfg.description_column].fillna("Desconhecido").astype(str)
	df[cfg.amount_column] = df[cfg.amount_column].astype(float)

	classification = df.apply(
		lambda row: _classify_transaction(
			row[cfg.description_column],
			row[cfg.amount_column],
			cfg,
		),
		axis=1,
		result_type="expand",
	)
	df["transaction_type"] = classification[0]
	df["signed_amount"] = classification[1]
	df["amount"] = df["signed_amount"]

	df["category"] = df[cfg.description_column].apply(cfg.category_for_description)
	df["is_subscription"] = df[cfg.description_column].str.lower().apply(cfg.is_subscription)
	df["abs_amount"] = df["amount"].abs()
	df["day_of_week"] = df[cfg.date_column].dt.dayofweek
	df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
	df["date_only"] = df[cfg.date_column].dt.date
	df["month"] = df[cfg.date_column].dt.to_period("M").dt.to_timestamp()
	df["year"] = df[cfg.date_column].dt.year
	df["hour"] = df[cfg.date_column].dt.hour.fillna(0).astype(int)
	df["merchant_clean"] = df[cfg.description_column].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True).str.strip()

	df = df.sort_values(cfg.date_column).reset_index(drop=True)
	df = df.drop(columns=["signed_amount"], errors="ignore")
	df["running_balance"] = df["amount"].cumsum()
	return df
