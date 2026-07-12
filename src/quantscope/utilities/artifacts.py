"""Run-directory artifact I/O with provenance-labeled metrics.

Every run writes into ``<output_dir>/<run_name>/``:

- ``config.json`` — the resolved experiment configuration
- ``environment.json`` — interpreter/package/device/git info
- ``metrics.json`` — metrics, each carrying a Provenance label (ADR-004)
- checkpoints and any extra files the workflow adds

Metrics are refused unless labeled: `record_metric` requires a
`Provenance` value, so nothing unlabeled can be persisted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from quantscope.config.schemas import ExperimentConfig, Provenance
from quantscope.utilities.reproducibility import capture_environment

__all__ = ["RunWriter", "read_metrics"]

logger = logging.getLogger(__name__)


class RunWriter:
    """Writes labeled artifacts for a single experiment run."""

    def __init__(self, config: ExperimentConfig, *, kind: str) -> None:
        """``kind`` names the workflow (e.g. "fp32", "ptq")."""
        self.config = config
        self.kind = kind
        self.run_dir = Path(config.output_dir) / f"{config.run_name}-{kind}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: list[dict[str, Any]] = []
        self._write_json("config.json", json.loads(config.model_dump_json()))
        self._write_json("environment.json", capture_environment())

    def _write_json(self, filename: str, payload: object) -> None:
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    def record_metric(
        self,
        name: str,
        value: float | int | dict[str, Any],
        provenance: Provenance,
        **context: Any,
    ) -> None:
        """Record one metric with a mandatory provenance label."""
        if not isinstance(provenance, Provenance):
            raise TypeError(
                f"provenance must be a Provenance label, got {type(provenance).__name__}"
            )
        entry: dict[str, Any] = {
            "name": name,
            "value": value,
            "provenance": provenance.value,
            **context,
        }
        self._metrics.append(entry)
        logger.info("metric %s=%s [%s]", name, value, provenance.value)

    def finalize(self) -> Path:
        """Write accumulated metrics and return the run directory."""
        self._write_json("metrics.json", {"kind": self.kind, "metrics": self._metrics})
        return self.run_dir


def read_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Load the metrics artifact from a run directory."""
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.json in {run_dir}")
    return json.loads(path.read_text())
