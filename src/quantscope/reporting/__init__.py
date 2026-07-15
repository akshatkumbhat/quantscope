"""Report generation: deterministic figures + manifest from run artifacts."""

from quantscope.reporting.build import build_report
from quantscope.reporting.report_data import (
    file_sha256,
    get_metric,
    load_labeled_metrics,
    load_observer_summary,
    load_sweep_records,
)

__all__ = [
    "build_report",
    "file_sha256",
    "get_metric",
    "load_labeled_metrics",
    "load_observer_summary",
    "load_sweep_records",
]
