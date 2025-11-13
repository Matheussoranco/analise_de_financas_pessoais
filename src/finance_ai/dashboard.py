"""Streamlit dashboard for the finance AI project."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st  # type: ignore

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from finance_ai.config import FinanceAIConfig, get_default_config
from finance_ai.pipeline import run_analysis


st.set_page_config(
    page_title="Finance AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Painel inteligente de financas pessoais")
st.caption("Analise automatizada de extratos de cartao de credito com IA.")


def _read_uploaded_files(files: List) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for uploaded in files:
        buffer = StringIO(uploaded.getvalue().decode("utf-8"))
        frame = pd.read_csv(buffer)
        frame["source_file"] = uploaded.name
        frames.append(frame)
    dataframe = pd.concat(frames, ignore_index=True)
    return dataframe


def main() -> None:
    cfg: FinanceAIConfig = get_default_config()
    st.sidebar.header("Fonte de dados")
    uploaded_files = st.sidebar.file_uploader(
        "Envie extratos Nubank em CSV",
        type=["csv"],
        accept_multiple_files=True,
    )
    use_local = st.sidebar.checkbox(
        f"Usar arquivos da pasta padrao ({cfg.raw_data_dir})",
        value=not uploaded_files,
    )

    dataframe = None
    sources = None
    if uploaded_files:
        dataframe = _read_uploaded_files(uploaded_files)
    elif use_local:
        sources = [str(path) for path in cfg.raw_data_dir.glob("*.csv")]

    if dataframe is None and not sources:
        st.info(
            "Adicione arquivos CSV ou marque a opcao para usar os dados existentes em data/."
        )
        return

    artifacts = run_analysis(sources=sources, dataframe=dataframe, config=cfg)

    metrics = artifacts.insights.cashflow_metrics
    cols = st.columns(3)
    cols[0].metric("Total de gastos", f"R$ {metrics['total_expense']:.2f}")
    cols[1].metric("Total de entradas", f"R$ {metrics['total_income']:.2f}")
    cols[2].metric("Fluxo liquido", f"R$ {metrics['net_cashflow']:.2f}")

    st.subheader("Destaques")
    for highlight in artifacts.insights.highlights:
        st.write(f"- {highlight}")

    st.subheader("Resumo mensal consolidado")
    monthly_summary = artifacts.quality.monthly_summary.copy()
    if monthly_summary.empty:
        st.write("Ainda nao ha informacoes consolidadas pelos meses.")
    else:
        monthly_summary["month"] = pd.to_datetime(monthly_summary["month"], errors="coerce")
        monthly_view = monthly_summary.sort_values("month")
        flagged = monthly_view.query("quality_flag == True")
        if not flagged.empty:
            parsed_months = pd.to_datetime(flagged["month"], errors="coerce")
            valid_months = parsed_months.dropna()
            month_labels = [month.strftime("%Y-%m") for month in valid_months]
        else:
            month_labels = []
        formatted = monthly_view.copy()
        formatted["month"] = formatted["month"].dt.strftime("%Y-%m")
        formatted["month"] = formatted["month"].replace("NaT", "")
        currency_cols = [
            "expense_sum",
            "income_sum",
            "net_cashflow",
            "largest_expense",
        ]
        for col in currency_cols:
            if col in formatted.columns:
                formatted[col] = formatted[col].round(2)
        score_cols = ["quality_score", "expense_per_transaction", "income_per_transaction"]
        for col in score_cols:
            if col in formatted.columns:
                formatted[col] = formatted[col].round(4)
        st.dataframe(formatted)
        if month_labels:
            st.warning(
                "Meses com possivel inconsistencia de dados: " + ", ".join(month_labels)
            )

    st.subheader("Categorias consolidadas")
    st.dataframe(artifacts.insights.category_breakdown)

    with st.expander("Detalhamento por categoria"):
        st.dataframe(artifacts.insights.category_breakdown)

    with st.expander("Assinaturas e recorrencias"):
        recurring = artifacts.insights.recurring_merchants
        if recurring.empty:
            st.write("Nenhuma recorrencia forte detectada.")
        else:
            st.dataframe(recurring)

    with st.expander("Gastos atipicos"):
        anomalies = artifacts.insights.anomaly_table
        if anomalies.empty:
            st.write("Sem anomalias relevantes.")
        else:
            st.dataframe(
                anomalies[
                    [
                        "date",
                        "merchant_clean",
                        "category",
                        "amount",
                        "anomaly_score",
                        "source_file",
                    ]
                ]
            )

    st.subheader("Transacoes processadas")
    processed_cols = [
        "date",
        "merchant_clean",
        "category",
        "amount",
        "transaction_type",
        "source_file",
    ]
    available_cols = [col for col in processed_cols if col in artifacts.anomalies.dataframe.columns]
    st.dataframe(
        artifacts.anomalies.dataframe.sort_values("date", ascending=False)[available_cols]
    )

    st.download_button(
        "Baixar dados processados (CSV)",
        data=artifacts.anomalies.dataframe.to_csv(index=False).encode("utf-8"),
        file_name="transacoes_processadas.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()