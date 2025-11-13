"""Anomaly detection for credit card transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import FinanceAIConfig, get_default_config


@dataclass(slots=True)
class AnomalyResult:
    dataframe: pd.DataFrame
    model: IsolationForest


class TransactionAnomalyDetector:
    def __init__(self, config: Optional[FinanceAIConfig] = None) -> None:
        self.config = config or get_default_config()
        params = self.config.anomaly
        self.model = IsolationForest(
            contamination=params.contamination,
            random_state=params.random_state,
        )

    def _build_feature_matrix(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        feature_columns = list(self.config.anomaly.feature_columns)
        features = dataframe.reindex(columns=feature_columns)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0.0)
        return features

    def score(self, dataframe: pd.DataFrame) -> AnomalyResult:
        df = dataframe.copy()
        feature_matrix = self._build_feature_matrix(df)
        labels = self.model.fit_predict(feature_matrix)
        scores = self.model.decision_function(feature_matrix)
        df["anomaly_score"] = scores
        df["is_anomaly"] = labels == -1
        return AnomalyResult(dataframe=df, model=self.model)


def detect_anomalies(
    dataframe: pd.DataFrame,
    *,
    config: FinanceAIConfig | None = None,
) -> AnomalyResult:
    detector = TransactionAnomalyDetector(config)
    return detector.score(dataframe)