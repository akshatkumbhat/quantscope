"""Command-line interface for QuantScope."""

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer

app = typer.Typer(
    name="quantscope",
    help="Hardware-aware neural-network quantization and numerical debugging.",
    no_args_is_help=True,
)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Hardware-aware neural-network quantization and numerical debugging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version_info() -> None:
    """Print the installed QuantScope version."""
    try:
        installed_version = version("quantscope")
    except PackageNotFoundError:
        installed_version = "development"

    typer.echo(f"QuantScope {installed_version}")


@app.command()
def env_info() -> None:
    """Print interpreter, package, and quantization-backend information."""
    from quantscope.utilities import capture_environment

    typer.echo(json.dumps(capture_environment(), indent=2))


@app.command()
def train_fp32(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    """Train the FP32 baseline described by an experiment config (measured)."""
    from quantscope.config import load_experiment_config
    from quantscope.evaluation.loop import train_fp32 as _train

    experiment = load_experiment_config(config)
    _, metrics = _train(experiment)
    typer.echo(f"fp32 eval accuracy (measured): {metrics['accuracy']:.4f}")


@app.command()
def ptq(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
) -> None:
    """Post-training-quantize the FP32 baseline via FX graph mode (measured INT8 CPU)."""
    from quantscope.config import load_experiment_config
    from quantscope.quantization.ptq import run_ptq

    experiment = load_experiment_config(config)
    _, metrics = run_ptq(experiment, checkpoint=checkpoint)
    typer.echo(f"int8 eval accuracy (measured, CPU): {metrics['accuracy']:.4f}")


@app.command()
def ablate(
    seed: int = typer.Option(0, "--seed"),
    runs_dir: str = typer.Option("runs/validation", "--runs-dir"),
    weight_bits: int = typer.Option(4, "--weight-bits"),
    act_bits: int = typer.Option(4, "--act-bits"),
    freq_step: float = typer.Option(0.30, "--freq-step"),
) -> None:
    """Per-group quantization ablation vs. an FP32 texture-bench checkpoint. Slow."""
    from quantscope.benchmark import benchmark_config, texture10_calibration
    from quantscope.data.synthetic import build_datasets
    from quantscope.quantization.simulate import SimQuantConfig
    from quantscope.sensitivity import ablate_groups

    config = benchmark_config(seed=seed, output_dir=runs_dir, freq_step=freq_step)
    checkpoint = Path(runs_dir) / f"texture-a-seed{seed}-fp32" / "model.pt"
    calibration = texture10_calibration(config)
    _, test_set = build_datasets(config.data, config.model)
    results = ablate_groups(
        config,
        calibration,
        test_set,
        checkpoint=checkpoint,
        target=SimQuantConfig(weight_bits, act_bits),
    )
    for group, deltas in sorted(results.items(), key=lambda kv: kv[1]["delta_nll"], reverse=True):
        typer.echo(
            f"{group:>14} (simulated): dNLL={deltas['delta_nll']:+.4f} "
            f"flips={deltas['prediction_flip_rate']:.4f} "
            f"dAcc={deltas['delta_accuracy']:+.4f}"
        )


@app.command()
def sweep(
    seed: int = typer.Option(0, "--seed"),
    runs_dir: str = typer.Option("runs/validation-012", "--runs-dir"),
    freq_step: float = typer.Option(0.12, "--freq-step"),
) -> None:
    """Exhaustive 256-config INT4/INT8 sweep vs an FP32 checkpoint. Very slow."""
    import json

    import torch

    from quantscope.benchmark import benchmark_config, texture10_calibration
    from quantscope.config import Provenance
    from quantscope.data.synthetic import build_datasets
    from quantscope.models.tiny_cnn import build_model
    from quantscope.search import exhaustive_sweep, pareto_frontier
    from quantscope.utilities import RunWriter

    config = benchmark_config(seed=seed, output_dir=runs_dir, freq_step=freq_step)
    checkpoint = Path(runs_dir) / f"texture-a-seed{seed}-fp32" / "model.pt"
    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    calibration = texture10_calibration(config)
    _, test_set = build_datasets(config.data, config.model)

    records = exhaustive_sweep(model, calibration, test_set, batch_size=config.training.batch_size)
    writer = RunWriter(config, kind="sweep")
    (writer.run_dir / "sweep_table.json").write_text(
        json.dumps([r.to_dict() for r in records], indent=1) + "\n"
    )
    frontier = pareto_frontier(records, quality="nll")
    writer.record_metric("num_configs", len(records), Provenance.SIMULATED)
    writer.record_metric(
        "pareto_size_nll",
        len(frontier),
        Provenance.SIMULATED,
        note="cost is estimated (normalized weight bits); nll simulated",
    )
    writer.finalize()
    typer.echo(f"sweep complete: {len(records)} configs, NLL-frontier size {len(frontier)}")


@app.command()
def backend_parity(
    seed: int = typer.Option(0, "--seed"),
    runs_dir: str = typer.Option("runs/validation-012", "--runs-dir"),
    freq_step: float = typer.Option(0.12, "--freq-step"),
) -> None:
    """Staged backend-parity ladder: graph-anchored sim vs reference-FX vs real INT8."""
    from quantscope.benchmark import benchmark_config, texture10_calibration
    from quantscope.data.synthetic import build_datasets
    from quantscope.quantization.parity import run_backend_parity

    config = benchmark_config(seed=seed, output_dir=runs_dir, freq_step=freq_step)
    checkpoint = Path(runs_dir) / f"texture-a-seed{seed}-fp32" / "model.pt"
    calibration = texture10_calibration(config)
    _, test_set = build_datasets(config.data, config.model)
    results = run_backend_parity(config, calibration, test_set, checkpoint=checkpoint)

    for stage in (
        "stage3_activation_only",
        "stage4_weight_only",
        "stage5_strict",
        "backend_comparison",
    ):
        r = results[stage]
        typer.echo(
            f"{stage}: disagree={r['prediction_disagreement']:.4f} "
            f"maxdiff={r['per_sample_max_absdiff_max']:.2e} sqnr={r['sqnr_db']:.1f}dB"
        )
    for name in ("sim_full", "reference_fx", "real_int8"):
        ev = results[f"{name}_eval"]
        typer.echo(f"{name}: accuracy={ev['accuracy']:.4f} nll={ev['nll']:.4f}")


@app.command()
def texture_bench(
    seed: int = typer.Option(0, "--seed"),
    epochs: int = typer.Option(35, "--epochs"),
    boundary_fraction: float = typer.Option(0.45, "--boundary-fraction"),
    boundary_low: float = typer.Option(0.40, "--boundary-low"),
    boundary_high: float = typer.Option(0.50, "--boundary-high"),
    snr_db: float = typer.Option(4.0, "--snr-db"),
    freq_step: float = typer.Option(0.30, "--freq-step"),
    bottleneck_width: int = typer.Option(6, "--bottleneck-width"),
    output_dir: str = typer.Option("runs", "--output-dir"),
) -> None:
    """Run Texture-10 benchmark A: FP32 (measured) vs simulated W8A8/W4A4. Slow."""
    from quantscope.benchmark import benchmark_config, run_texture_benchmark

    config = benchmark_config(
        seed=seed,
        epochs=epochs,
        boundary_fraction=boundary_fraction,
        boundary_low=boundary_low,
        boundary_high=boundary_high,
        snr_db=snr_db,
        freq_step=freq_step,
        bottleneck_width=bottleneck_width,
        output_dir=output_dir,
    )
    results = run_texture_benchmark(config)
    for label, metrics in results.items():
        provenance = "measured" if label == "fp32" else "simulated"
        typer.echo(
            f"{label:>5} ({provenance}): accuracy={metrics['accuracy']:.4f} "
            f"nll={metrics['nll']:.4f} margin={metrics['mean_margin']:.3f}"
        )


@app.command()
def hw_validate(
    profile: Path = typer.Option("configs/hardware/generic_edge_npu.yaml", "--profile"),
) -> None:
    """Validate a schema-v1 hardware profile and print its summary (estimated model)."""
    from quantscope.hardware import load_hardware_profile

    loaded = load_hardware_profile(profile)
    p = loaded.profile
    typer.echo(f"profile: {p.profile_name} (schema v{p.schema_version}, fictional={p.fictional})")
    typer.echo(f"source sha256:    {loaded.source_sha256}")
    typer.echo(f"canonical digest: {loaded.canonical_digest}")
    typer.echo(f"unit: {p.unit}")
    typer.echo(f"traffic model: {p.traffic_model}")
    for entry in p.compute_ncu_per_mac:
        typer.echo(f"  compute {entry.label}: {entry.ncu} ncu/MAC (estimated)")
    typer.echo(
        f"  memory: weights {p.weight_memory_ncu_per_bit} ncu/bit, "
        f"activations {p.activation_memory_ncu_per_bit} ncu/bit, "
        f"overhead {p.per_layer_overhead_ncu} ncu/layer (estimated)"
    )
    typer.echo("VALID")


@app.command()
def hw_score(
    bits: str = typer.Option(..., "--bits", help="8 per-group bits, e.g. 4,8,8,4,4,8,8,4"),
    profile: Path = typer.Option("configs/hardware/generic_edge_npu.yaml", "--profile"),
) -> None:
    """Component-wise ESTIMATED cost of one mixed-precision assignment."""
    from quantscope.benchmark import benchmark_config
    from quantscope.hardware import (
        account_model,
        config_identifier,
        configuration_cost,
        load_hardware_profile,
    )
    from quantscope.models.tiny_cnn import build_model

    loaded = load_hardware_profile(profile)
    values = [int(b) for b in bits.split(",")]
    assignment = [(b, b) for b in values]  # B3 semantics: each group W4A4 or W8A8
    accounting = account_model(build_model(benchmark_config(seed=0).model))
    cost = configuration_cost(accounting, assignment, loaded.profile)
    baseline = configuration_cost(accounting, [(8, 8)] * len(values), loaded.profile)
    typer.echo(f"configuration: {config_identifier(assignment)}")
    for name, comp in cost.per_group.items():
        typer.echo(
            f"  {name:<15} compute {comp.compute:12.1f}  wmem {comp.weight_memory:10.1f}  "
            f"amem {comp.activation_memory:10.1f}  total {comp.total:12.1f} ncu (estimated)"
        )
    typer.echo(
        f"total {cost.total:.1f} ncu (estimated); normalized vs all-INT8 "
        f"{cost.total / baseline.total:.4f} — modeled quantizable workload only"
    )


regression_app = typer.Typer(
    name="regression",
    help="Numerical-regression harness (ADR-015). Exit codes: 0 pass, "
    "1 regression, 2 malformed/incompatible input.",
    no_args_is_help=True,
)
app.add_typer(regression_app, name="regression")


def _load_artifact(path: Path) -> dict:
    import json

    from quantscope.regression import HarnessError

    if not path.exists():
        raise HarnessError(f"artifact not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise HarnessError(f"artifact {path} is not valid JSON: {error}") from error


def _harness_exit(error: Exception) -> None:
    typer.echo(f"HARNESS ERROR (exit 2): {error}", err=True)
    raise typer.Exit(2)


@regression_app.command("smoke")
def regression_smoke(out: Path = typer.Option(..., "--out")) -> None:
    """Generate the deterministic offline smoke artifact (no training)."""
    from quantscope.regression import write_smoke_artifact

    write_smoke_artifact(out)
    typer.echo(f"smoke artifact: {out}")


@regression_app.command("validate-baseline")
def regression_validate_baseline(baseline: Path = typer.Argument(...)) -> None:
    """Validate a baseline file (schema, canonical digest, rule uniqueness)."""
    from quantscope.regression import HarnessError, load_baseline

    try:
        spec = load_baseline(baseline)
    except HarnessError as error:
        _harness_exit(error)
    typer.echo(
        f"baseline {spec.baseline_name!r}: {len(spec.rules)} rules, digest "
        f"{spec.canonical_digest[:12]}… VALID"
    )


@regression_app.command("check")
def regression_check(
    artifact: Path = typer.Argument(...),
    baseline: Path = typer.Option(..., "--baseline"),
    diff_out: Path | None = typer.Option(None, "--diff-out"),
) -> None:
    """Check an artifact against a baseline. Never updates baselines."""
    from quantscope.regression import HarnessError, check_artifact, load_baseline, write_diff

    try:
        spec = load_baseline(baseline)
        report = check_artifact(_load_artifact(artifact), spec)
    except HarnessError as error:
        _harness_exit(error)
    if diff_out is not None:
        write_diff(report, diff_out)
    if report.passed:
        typer.echo(f"PASS: {len(report.entries)} checks against {spec.baseline_name!r}")
        return
    counts = ", ".join(f"{k}: {v}" for k, v in sorted(report.failure_counts().items()))
    typer.echo(f"REGRESSION (exit 1): {len(report.failures)} failed checks ({counts})")
    for entry in report.failures[:5]:
        typer.echo(
            f"  {entry.path}: expected {entry.expected!r}, got {entry.actual!r} "
            f"({entry.explanation})"
        )
    if len(report.failures) > 5:
        typer.echo(f"  … {len(report.failures) - 5} more")
    if diff_out is not None:
        typer.echo(f"full structured diff: {diff_out}")
    raise typer.Exit(1)


@regression_app.command("capture")
def regression_capture(
    artifact: Path = typer.Argument(...),
    output: Path = typer.Option(..., "--output"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Propose a baseline from an artifact (a code-review decision; never CI)."""
    import json

    from quantscope.regression import HarnessError, capture_baseline

    try:
        spec, comparison = capture_baseline(_load_artifact(artifact), output, overwrite=overwrite)
    except HarnessError as error:
        _harness_exit(error)
    typer.echo(f"proposed baseline: {output} ({len(spec.rules)} rules) — review before commit")
    if comparison is not None:
        typer.echo("comparison with previous baseline:")
        typer.echo(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
