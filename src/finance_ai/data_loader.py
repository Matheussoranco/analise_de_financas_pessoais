"""Data access helpers for finance statements."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .config import FinanceAIConfig, get_default_config

logger = logging.getLogger(__name__)


def _resolve_sources(
    sources: Optional[Sequence[Path | str]],
    config: FinanceAIConfig,
) -> List[Path]:
    if sources:
        resolved = [Path(src).expanduser().resolve() for src in sources]
    else:
        resolved = sorted(config.raw_data_dir.glob("*.csv"))
    if not resolved:
        msg = (
            "Nenhum arquivo CSV encontrado. Coloque seus extratos em "
            f"{config.raw_data_dir} ou informe os caminhos manualmente."
        )
        raise FileNotFoundError(msg)
    return resolved


def load_transactions(
    sources: Optional[Sequence[Path | str]] = None,
    *,
    config: Optional[FinanceAIConfig] = None,
) -> pd.DataFrame:
    """Load Nubank-like CSV statements into a normalized dataframe."""

    cfg = config or get_default_config()
    cfg.ensure_directories()
    csv_files = _resolve_sources(sources, cfg)
    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        logger.info("Loading transactions from %s", csv_file)
        frame = pd.read_csv(
            csv_file,
            sep=",",
            parse_dates=[cfg.date_column],
            dayfirst=False,
            dtype={cfg.description_column: "string"},
        )
        frame[cfg.date_column] = pd.to_datetime(frame[cfg.date_column], errors="coerce")
        frame[cfg.description_column] = frame[cfg.description_column].fillna("Desconhecido")
        frame[cfg.amount_column] = (
            frame[cfg.amount_column]
            .astype(str)
            .str.replace(".", ".", regex=False)
            .str.replace(",", ".")
            .astype(float)
        )
        frame["source_file"] = csv_file.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=[cfg.date_column, cfg.description_column, cfg.amount_column])
    combined = combined.sort_values(cfg.date_column).reset_index(drop=True)
    return combined


def save_processed_dataset(
    dataframe: pd.DataFrame,
    filename: str,
    *,
    config: Optional[FinanceAIConfig] = None,
) -> Path:
    cfg = config or get_default_config()
    cfg.ensure_directories()
    target_path = cfg.processed_data_dir / filename
    dataframe.to_parquet(target_path, index=False)
    logger.info("Processed dataset stored at %s", target_path)
    return target_path


def list_available_sources(config: Optional[FinanceAIConfig] = None) -> Iterable[Path]:
    cfg = config or get_default_config()
    return sorted(cfg.raw_data_dir.glob("*.csv"))
