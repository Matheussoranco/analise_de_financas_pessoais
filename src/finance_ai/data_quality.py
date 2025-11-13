"""Monthly data-quality assessment using robust machine learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from .config import FinanceAIConfig, get_default_config


@dataclass(slots=True)
class DataQualityResult:
    """Container for the outcome of the quality assessment."""

    monthly_summary: pd.DataFrame
    model: Optional[Pipeline]
    feature_names: List[str]
    threshold: float


class MonthlyDataQualityModel:
    """Model that learns the typical structure of monthly statements."""

    def __init__(self, config: Optional[FinanceAIConfig] = None) -> None:
        self.config = config or get_default_config()
        params = self.config.quality
        self.pipeline = Pipeline(
            steps=[
                ("scale", RobustScaler()),
                ("model", OneClassSVM(gamma=params.gamma, nu=params.nu)),
            ]
        )
        self.threshold = params.score_threshold

    def _ensure_month_column(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if "month" in dataframe.columns:
            return dataframe
        df = dataframe.copy()
        df["month"] = pd.to_datetime(df[self.config.date_column], errors="coerce").dt.to_period("M").dt.to_timestamp()
        return df

    def _build_monthly_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_month_column(dataframe)
        df = df.copy()
        df["expense_value"] = np.where(df["transaction_type"] == "expense", -df["amount"], 0.0)
        df["income_value"] = np.where(df["transaction_type"] != "expense", df["amount"], 0.0)
        df["refund_flag"] = np.where(df["transaction_type"] == "refund", 1.0, 0.0)
        if "is_subscription" in df.columns:
            df["subscription_flag"] = df["is_subscription"].astype(float)
        else:
            df["subscription_flag"] = np.zeros(len(df), dtype=float)
        if "is_weekend" in df.columns:
            df["weekend_flag"] = df["is_weekend"].astype(float)
        else:
            df["weekend_flag"] = np.zeros(len(df), dtype=float)
        if "is_anomaly" in df.columns:
            df["anomaly_flag"] = df["is_anomaly"].astype(float)
        else:
            df["anomaly_flag"] = np.zeros(len(df), dtype=float)
        df["missing_amount_flag"] = df[self.config.amount_column].isna().astype(float)

        monthly = (
            df.groupby("month")
            .agg(
                total_transactions=("amount", "size"),
                expense_sum=("expense_value", "sum"),
                income_sum=("income_value", "sum"),
                avg_expense=("expense_value", "mean"),
                std_expense=("expense_value", "std"),
                avg_income=("income_value", "mean"),
                std_income=("income_value", "std"),
                avg_abs_amount=("abs_amount", "mean"),
                std_abs_amount=("abs_amount", "std"),
                weekend_ratio=("weekend_flag", "mean"),
                subscription_ratio=("subscription_flag", "mean"),
                refund_ratio=("refund_flag", "mean"),
                anomaly_ratio=("anomaly_flag", "mean"),
                missing_amount_ratio=("missing_amount_flag", "mean"),
                largest_expense=("expense_value", "max"),
            )
            .sort_index()
        )
        if monthly.empty:
            return monthly
        monthly["net_cashflow"] = monthly["income_sum"] - monthly["expense_sum"]
        monthly["expense_per_transaction"] = monthly["expense_sum"] / monthly["total_transactions"].clip(lower=1)
        monthly["income_per_transaction"] = monthly["income_sum"] / monthly["total_transactions"].clip(lower=1)
        monthly = monthly.fillna(0.0)
        return monthly

    def score(self, dataframe: pd.DataFrame) -> DataQualityResult:
        monthly_features = self._build_monthly_features(dataframe)
        feature_names = list(monthly_features.columns)
        if monthly_features.empty:
            summary = monthly_features.copy()
            summary["quality_score"] = np.nan
            summary["quality_flag"] = False
            summary = summary.reset_index()
            return DataQualityResult(summary, None, feature_names, self.threshold)

        params = self.config.quality
        if len(monthly_features) < params.min_months:
            summary = monthly_features.copy()
            summary["quality_score"] = 0.0
            summary["quality_flag"] = False
            summary = summary.reset_index()
            return DataQualityResult(summary, None, feature_names, self.threshold)

        pipeline = self.pipeline
        pipeline.fit(monthly_features)
        scores = pipeline.decision_function(monthly_features)
        summary = monthly_features.copy()
        summary["quality_score"] = scores
        summary["quality_flag"] = summary["quality_score"] < self.threshold
        summary = summary.reset_index()
        return DataQualityResult(summary, pipeline, feature_names, self.threshold)


def assess_data_quality(
    dataframe: pd.DataFrame,
    *,
    config: Optional[FinanceAIConfig] = None,
) -> DataQualityResult:
    model = MonthlyDataQualityModel(config)
    return model.score(dataframe)
