"""Finance AI package entry-point exports."""

from typing import TYPE_CHECKING

from .config import FinanceAIConfig, get_default_config

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
	from .pipeline import AnalysisArtifacts as _AnalysisArtifacts


def run_analysis(*args, **kwargs):
	from .pipeline import run_analysis as _run_analysis

	return _run_analysis(*args, **kwargs)


if TYPE_CHECKING:  # pragma: no cover
	AnalysisArtifacts = _AnalysisArtifacts
else:
	AnalysisArtifacts = object  # placeholder for consumers without importing pipeline


__all__ = ["FinanceAIConfig", "get_default_config", "run_analysis", "AnalysisArtifacts"]
