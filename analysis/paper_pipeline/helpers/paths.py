from __future__ import annotations

from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parents[1]
ANALYSES_DIR = PIPELINE_ROOT / "analyses"
DATA_DIR = PIPELINE_ROOT / "data"
FIGURES_DIR = PIPELINE_ROOT / "figures"


def analysis_data_dir(analysis_slug: str) -> Path:
    """Return and create the standard data directory for one analysis."""
    path = DATA_DIR / analysis_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_figure_dir(analysis_slug: str) -> Path:
    """Return and create the standard figure directory for one analysis."""
    path = FIGURES_DIR / analysis_slug
    path.mkdir(parents=True, exist_ok=True)
    return path

