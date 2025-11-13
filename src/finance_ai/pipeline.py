"""End-to-end orchestration for the finance AI workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

try:  # pragma: no cover - import guard for script execution
    from .anomaly_detection import AnomalyResult, detect_anomalies
    from .config import FinanceAIConfig, get_default_config
    from .data_loader import load_transactions
    from .data_quality import DataQualityResult, assess_data_quality
    from .feature_engineering import engineer_features
    from .forecasting import ForecastResult, forecast_expenses
    from .insights import InsightReport, generate_insight_report
    from .preprocessing import prepare_transactions
except ImportError:  # executed as a stand-alone script
    import sys
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT.parent) not in sys.path:
        sys.path.append(str(PACKAGE_ROOT.parent))

    from finance_ai.anomaly_detection import AnomalyResult, detect_anomalies  # type: ignore
    from finance_ai.config import FinanceAIConfig, get_default_config  # type: ignore
    from finance_ai.data_loader import load_transactions  # type: ignore
    from finance_ai.data_quality import DataQualityResult, assess_data_quality  # type: ignore
    from finance_ai.feature_engineering import engineer_features  # type: ignore
    from finance_ai.forecasting import ForecastResult, forecast_expenses  # type: ignore
    from finance_ai.insights import InsightReport, generate_insight_report  # type: ignore
    from finance_ai.preprocessing import prepare_transactions  # type: ignore


@dataclass(slots=True)
class AnalysisArtifacts:
    raw: pd.DataFrame
    processed: pd.DataFrame
    features: pd.DataFrame
    anomalies: AnomalyResult
    quality: DataQualityResult
    forecast: ForecastResult
    insights: InsightReport


def run_analysis(
    sources: Optional[Sequence[str]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    *,
    config: Optional[FinanceAIConfig] = None,
) -> AnalysisArtifacts:
    cfg = config or get_default_config()
    if dataframe is not None:
        raw = dataframe.copy()
    else:
        raw = load_transactions(sources, config=cfg)
    processed = prepare_transactions(raw, config=cfg)
    features = engineer_features(processed, config=cfg)
    anomalies = detect_anomalies(features, config=cfg)
    quality = assess_data_quality(anomalies.dataframe, config=cfg)
    forecast = forecast_expenses(anomalies.dataframe, config=cfg)
    insights = generate_insight_report(
        anomalies.dataframe,
        forecast,
        quality=quality,
        config=cfg,
    )
    return AnalysisArtifacts(
        raw=raw,
        processed=processed,
        features=features,
        anomalies=anomalies,
        quality=quality,
        forecast=forecast,
        insights=insights,
    )


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa a analise financeira completa.")
    parser.add_argument(
        "paths",
        metavar="CSV",
        nargs="*",
        help="Arquivos CSV para carregar. Se vazio, busca em data/",
    )
    parser.add_argument(
        "--export",
        dest="export",
        metavar="ARQUIVO",
        help="Exporta as transacoes processadas para CSV",
    )
    return parser


def _render_console_report(artifacts: AnalysisArtifacts) -> None:
    insights = artifacts.insights
    print(insights.headline)
    for highlight in insights.highlights:
        print(f" - {highlight}")
    print("\nTop categorias:")
    print(insights.category_breakdown.head(5).to_string(index=False))
    if not insights.quality_summary.empty:
        display_cols = [
            "month",
            "total_transactions",
            "expense_sum",
            "income_sum",
            "quality_score",
            "quality_flag",
        ]
        available = [col for col in display_cols if col in insights.quality_summary.columns]
        if available:
            print("\nResumo mensal processado:")
            print(insights.quality_summary[available].to_string(index=False))
    if not insights.anomaly_table.empty:
        print("\nGastos atipicos detectados:")
        print(
            insights.anomaly_table[
                ["date", "merchant_clean", "category", "amount", "anomaly_score"]
            ]
            .head(10)
            .to_string(index=False)
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _create_argument_parser()
    args = parser.parse_args(argv)
    artifacts = run_analysis(args.paths or None)
    _render_console_report(artifacts)
    if args.export:
        export_path = args.export
        artifacts.anomalies.dataframe.to_csv(export_path, index=False)
        print(f"\nArquivo processado salvo em {export_path}")


if __name__ == "__main__":
    main()