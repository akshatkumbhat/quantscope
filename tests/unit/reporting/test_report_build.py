"""Unit tests for the report builder: artifact loading, provenance
labels, fail-loud behavior, and byte-deterministic outputs.

Uses small synthetic artifacts in tmp_path; never touches runs/.
"""

import json
from pathlib import Path

import pytest

from quantscope.reporting import build_report, get_metric, load_labeled_metrics

SEED = 0
OBSERVERS = ("minmax", "percentile", "mse_grid", "pow2")
CONFIGS = ("W4A4", "W8A4", "W8A8")
STAGES = ("input", "input+early", "input+early+deeper")


def _make_validation_dir(base: Path) -> Path:
    validation = base / "validation"

    sweep_dir = validation / f"texture-a-seed{SEED}-sweep"
    sweep_dir.mkdir(parents=True)
    records = [
        {
            "bits": [4, 8, 4, 8, 4, 8, 4, 8],
            "cost": 0.5 + 0.05 * i,
            "nll": 0.20 - 0.01 * i,
            "accuracy": 0.90,
        }
        for i in range(6)
    ]
    (sweep_dir / "sweep_table.json").write_text(json.dumps(records))

    study_dir = validation / f"texture-a-seed{SEED}-observer-study"
    study_dir.mkdir(parents=True)
    metrics = [
        {"name": "fp32[clean_eval][nll]", "value": 0.11, "provenance": "measured"},
    ]
    for i, obs in enumerate(OBSERVERS):
        for j, cond in enumerate(("clean->clean", "stressed->clean")):
            metrics.append(
                {
                    "name": f"{obs}|W4A4|{cond}",
                    "value": {"nll": 0.15 + 0.02 * i + 0.05 * j, "accuracy": 0.93},
                    "provenance": "simulated",
                }
            )
    (study_dir / "metrics.json").write_text(
        json.dumps({"kind": "observer-study", "metrics": metrics})
    )

    summary = {
        "Q1_robustness_primary_condition": {},
        "Q2_clean_non_inferiority": {},
        "Q3_pow2_cost_measurement_only": {
            cfg: {
                cond: {f"seed{SEED}_nll_cost_vs_minmax": 0.01 * (k + 1)}
                for k, cond in enumerate(("clean->clean", "stressed->clean"))
            }
            for cfg in CONFIGS
        },
        "mechanism_decomposition": {
            f"seed{SEED}": {
                "total_damage": 0.06,
                "cumulative_damage": {s: 0.05 + 0.005 * i for i, s in enumerate(STAGES)},
            }
        },
    }
    (validation / "summary.json").write_text(json.dumps(summary))
    return validation


class TestBuildReport:
    def test_builds_figures_and_manifest(self, tmp_path: Path) -> None:
        validation = _make_validation_dir(tmp_path)
        out = tmp_path / "out"
        manifest_path = build_report(validation, validation / "summary.json", out, seeds=(SEED,))
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["figures"]) == 4
        for entry in manifest["figures"]:
            assert (out / entry["output"]).exists()
            assert entry["output_sha256"]
            assert entry["provenance"], f"figure {entry['figure']} has no provenance labels"
            for src in entry["sources"]:
                assert Path(src["path"]).exists()
                assert len(src["sha256"]) == 64

    def test_field_level_provenance_labels(self, tmp_path: Path) -> None:
        validation = _make_validation_dir(tmp_path)
        out = tmp_path / "out"
        manifest = json.loads(
            build_report(validation, validation / "summary.json", out, seeds=(SEED,)).read_text()
        )
        by_name = {e["figure"]: e for e in manifest["figures"]}
        # Pareto: task metric simulated, analytical cost estimated —
        # never a whole-figure "measured" claim.
        assert by_name["pareto_frontiers"]["provenance"] == {
            "nll": "simulated",
            "cost": "estimated",
        }
        assert by_name["observer_factorial_w4a4"]["provenance"] == {
            "quantized_nll": "simulated",
            "fp32_reference": "measured",
        }
        assert by_name["mechanism_decomposition"]["provenance"] == {"delta_nll": "simulated"}
        assert by_name["pow2_cost"]["provenance"] == {"nll_cost": "simulated"}
        legend = manifest["provenance_legend"]
        assert set(legend) == {"measured", "simulated", "estimated"}

    def test_outputs_are_deterministic(self, tmp_path: Path) -> None:
        validation = _make_validation_dir(tmp_path)
        first = build_report(validation, validation / "summary.json", tmp_path / "a", seeds=(SEED,))
        second = build_report(
            validation, validation / "summary.json", tmp_path / "b", seeds=(SEED,)
        )
        assert first.read_text() == second.read_text()
        for png in sorted((tmp_path / "a").glob("*.png")):
            assert png.read_bytes() == (tmp_path / "b" / png.name).read_bytes(), png.name

    def test_missing_artifact_fails_loudly(self, tmp_path: Path) -> None:
        validation = _make_validation_dir(tmp_path)
        (validation / f"texture-a-seed{SEED}-sweep" / "sweep_table.json").unlink()
        with pytest.raises(FileNotFoundError, match=r"sweep_table\.json"):
            build_report(validation, validation / "summary.json", tmp_path / "out", seeds=(SEED,))

    def test_missing_field_fails_loudly(self, tmp_path: Path) -> None:
        validation = _make_validation_dir(tmp_path)
        study = validation / f"texture-a-seed{SEED}-observer-study" / "metrics.json"
        payload = json.loads(study.read_text())
        payload["metrics"] = [m for m in payload["metrics"] if m["name"] != "fp32[clean_eval][nll]"]
        study.write_text(json.dumps(payload))
        with pytest.raises(KeyError, match=r"fp32\[clean_eval\]\[nll\]"):
            build_report(validation, validation / "summary.json", tmp_path / "out", seeds=(SEED,))

    def test_missing_summary_section_fails_loudly(self, tmp_path: Path) -> None:
        validation = _make_validation_dir(tmp_path)
        summary_path = validation / "summary.json"
        summary = json.loads(summary_path.read_text())
        del summary["mechanism_decomposition"]
        summary_path.write_text(json.dumps(summary))
        with pytest.raises(ValueError, match="mechanism_decomposition"):
            build_report(validation, summary_path, tmp_path / "out", seeds=(SEED,))


class TestLabeledMetricsLoader:
    def test_unlabeled_metric_rejected(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        run.mkdir()
        (run / "metrics.json").write_text(
            json.dumps({"kind": "x", "metrics": [{"name": "m", "value": 1.0}]})
        )
        with pytest.raises(ValueError, match="provenance"):
            load_labeled_metrics(run)

    def test_get_metric_names_missing_field(self) -> None:
        with pytest.raises(KeyError, match="absent"):
            get_metric({"present": {"value": 1, "provenance": "measured"}}, "absent")
