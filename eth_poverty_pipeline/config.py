"""Central configuration for the pipeline.

Paths resolve relative to the package location, not the current working
directory, so behavior is identical regardless of where the pipeline is
invoked from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

# The real data lives in a sibling repo (../DHS), gitignored there per the
# World Bank LSMS-ISA terms of use (public-use, free account required -
# see DHS/docs/wb_lsms_data_checklist.md). This pipeline never bundles or
# commits that data; it reads it from this path if present, and falls back
# to a synthetic panel with a matching schema otherwise (see
# data_extraction/loader.py).
DEFAULT_REAL_DATA_PATH = os.path.join(
    os.path.dirname(REPO_ROOT), "DHS", "data", "processed", "household_wave_panel_i.parquet"
)


@dataclass
class PipelineConfig:
    """Runtime parameters for a pipeline run.

    ``seed`` drives a local ``numpy.random.Generator`` (threaded through
    every stage) rather than the global ``numpy.random`` state.
    """

    seed: int = 42
    real_data_path: str = field(default=DEFAULT_REAL_DATA_PATH)
    validation_fraction: float = 0.20

    # A household-wave is "poor" if its nom_totcons_aeq falls in the
    # bottom `poverty_quantile` of its own wave (relative, within-wave
    # poverty - consumption is nominal birr, not inflation-adjusted across
    # 2011/2013/2015, so an absolute cross-wave threshold isn't valid).
    poverty_quantile: float = 1.0 / 3.0

    # Mock annotator's target accuracy (used only when client="mock").
    mock_accuracy: float = 0.80

    n_bootstrap: int = 300

    output_dir: str = field(default=OUTPUT_DIR)

    def ensure_output_dir(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir
