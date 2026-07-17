"""Deterministic offline smoke artifact + its baseline builder (ADR-015).

No training, no network, no historical run directories: a seeded
untrained TinyCNN evaluated on fixed synthetic data (determinism is
the regression target, not task quality), simulated W8A8 and W4A4
metrics with representative qparams and saturation diagnostics, and
one estimated hardware-cost value from the benchmark model accounting
(structure-only; needs no trained weights).
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantscope.data.synthetic import make_synthetic_dataset
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.models.tiny_cnn import TinyCNN
from quantscope.quantization.affine import quantize
from quantscope.quantization.simulate import (
    SimQuantConfig,
    calibrate_activation_params,
    quantize_weights_uniform,
    simulate_quantized,
)
from quantscope.regression.diff import atomic_write_json
from quantscope.regression.models import (
    BASELINE_SCHEMA_VERSION,
    BaselineSpec,
    CheckRule,
    ComparatorType,
    EnvironmentRules,
)
from quantscope.sensitivity import predictions

__all__ = ["build_smoke_baseline", "generate_smoke_artifact", "write_smoke_artifact"]

ARTIFACT_TYPE = "regression-smoke"
ARTIFACT_SCHEMA_VERSION = 1
SEED = 0
NUM_EVAL = 64
NUM_CALIB = 32
CONFIGS = (SimQuantConfig(8, 8), SimQuantConfig(4, 4))

# Tolerance families (ADR-015 table; rationales attached per rule).
_TORCH_FLOAT = {
    "atol": 1e-6,
    "rtol": 1e-5,
    "rationale": "float32 torch reductions may reorder across CPU/BLAS builds within ulps",
}
_NUMPY_SCALE = {
    "atol": 1e-9,
    "rtol": 1e-7,
    "rationale": "scales stored float32; headroom only for libm-level platform variation",
}
# ReLU-site scales are min-max extremes of activations computed THROUGH
# float32 torch convolutions: cross-BLAS builds differ by a few ulps
# (~1e-7 relative, observed on Linux CI vs the macOS capture), so they
# take the torch family, quantified before widening (ADR-015).
_TORCH_SCALE = {
    "atol": 1e-9,
    "rtol": 1e-5,
    "rationale": "activation extremes pass through float32 torch convs; cross-BLAS "
    "differences of a few ulps (~1e-7 rel observed) with 1e-5 headroom",
}
_HW_COST = {
    "atol": 1e-12,
    "rtol": 0.0,
    "rationale": "pure float64 arithmetic on integer counts times declared coefficients",
}


def _environment() -> dict[str, str]:
    try:
        qs_version = version("quantscope")
    except PackageNotFoundError:
        qs_version = "development"
    v = sys.version_info
    return {
        "python": f"{v.major}.{v.minor}.{v.micro}",
        "python_minor": f"{v.major}.{v.minor}",
        # Normalized (build suffixes like "+cpu" stripped) so the exact
        # environment gate compares versions, not wheel variants; the
        # full string stays report-only.
        "torch": torch.__version__.split("+")[0],
        "torch_full": torch.__version__,
        "numpy": np.__version__,
        "quantscope": qs_version,
        "platform": platform.platform(),
    }


def _leaf(value: Any, provenance: str) -> dict:
    return {"value": value, "provenance": provenance}


def generate_smoke_artifact() -> dict:
    """Build the deterministic smoke artifact (pure function of seeds)."""
    torch.manual_seed(SEED)
    model = TinyCNN(num_classes=4, in_channels=1).eval()
    eval_set = make_synthetic_dataset(
        num_samples=NUM_EVAL, seed=SEED + 1, image_size=16, num_classes=4, in_channels=1
    )
    calib = make_synthetic_dataset(
        num_samples=NUM_CALIB, seed=SEED + 2, image_size=16, num_classes=4, in_channels=1
    )
    labels = eval_set.tensors[1].numpy()

    def _eval_section(net, provenance: str) -> dict:
        detail = evaluate_detailed(net, eval_set)
        correct = int((predictions(net, eval_set) == labels).sum())
        return {
            "sample_count": _leaf(NUM_EVAL, provenance),
            "correct_count": _leaf(correct, provenance),
            "nll": _leaf(detail["nll"], provenance),
            "mean_margin": _leaf(detail["mean_margin"], provenance),
        }

    sections: dict[str, Any] = {"fp32": _eval_section(model, "measured")}
    for cfg in CONFIGS:
        sim = simulate_quantized(model, calib, cfg)
        section = _eval_section(sim, "simulated")
        act_params = calibrate_activation_params(
            quantize_weights_uniform(model, bits=cfg.weight_bits), calib, bits=cfg.act_bits
        )
        section["scales"] = {
            site: _leaf(float(np.asarray(p.scale)), "measured") for site, p in act_params.items()
        }
        section["zero_points"] = {
            site: _leaf(int(np.asarray(p.zero_point)), "measured") for site, p in act_params.items()
        }
        saturation = {}
        calib_images = calib.tensors[0].numpy()
        for site, p in act_params.items():
            if site == "__input__":
                codes = quantize(calib_images, p)
            else:
                continue  # input site is representative; ReLU capture stays out of smoke
            saturated = int(np.count_nonzero(codes <= p.qmin) + np.count_nonzero(codes >= p.qmax))
            saturation[site] = _leaf(saturated / codes.size, "simulated")
        section["saturation"] = saturation
        sections[cfg.label.lower()] = section

    # Estimated hardware cost: benchmark-model accounting (structure
    # only, untrained weights are irrelevant to MACs/elements).
    from quantscope.benchmark import benchmark_config
    from quantscope.hardware import account_model, configuration_cost, load_hardware_profile
    from quantscope.models.tiny_cnn import build_model

    loaded = load_hardware_profile("configs/hardware/generic_edge_npu.yaml")
    accounting = account_model(build_model(benchmark_config(seed=0).model))
    groups = len(accounting.groups)
    all4 = configuration_cost(accounting, [(4, 4)] * groups, loaded.profile)
    all8 = configuration_cost(accounting, [(8, 8)] * groups, loaded.profile)
    sections["hardware_cost"] = {
        "profile_digest": loaded.canonical_digest,
        "accounting_digest": accounting.digest(),
        "all_int4_normalized": _leaf(all4.total / all8.total, "estimated"),
    }

    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "identifiers": {
            "model": "tiny_cnn-4c-16px-seed0",
            "quantization_policy": "sim-policy-v1",
            "qparam_policy": "quantscope",
            "hardware_profile": "generic_edge_npu",
        },
        "environment": _environment(),
        "sections": sections,
    }


def write_smoke_artifact(path: str | Path) -> dict:
    artifact = generate_smoke_artifact()
    atomic_write_json(Path(path), artifact)
    return artifact


def _element_count(shape_elems: int) -> float:
    return 0.5 / shape_elems


def build_smoke_baseline(artifact: dict, *, capture_command: str) -> BaselineSpec:
    """Apply the preregistered per-family tolerance policy to a smoke
    artifact, producing a proposed baseline (still needs review)."""
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"capture supports artifact_type {ARTIFACT_TYPE!r} in v1")
    sections = artifact["sections"]
    rules: list[CheckRule] = [
        CheckRule(
            path="/sections",
            comparator=ComparatorType.STRUCTURE,
            required_keys=sorted(sections),
        ),
    ]
    for key, value in sorted(artifact["identifiers"].items()):
        rules.append(
            CheckRule(path=f"/identifiers/{key}", comparator=ComparatorType.EXACT, expected=value)
        )
    for name in ("profile_digest", "accounting_digest"):
        rules.append(
            CheckRule(
                path=f"/sections/hardware_cost/{name}",
                comparator=ComparatorType.EXACT,
                expected=sections["hardware_cost"][name],
            )
        )
    rules.append(
        CheckRule(
            path="/sections/hardware_cost/all_int4_normalized",
            comparator=ComparatorType.CLOSE,
            expected=sections["hardware_cost"]["all_int4_normalized"]["value"],
            provenance="estimated",
            **_HW_COST,
        )
    )
    for section in ("fp32", "w8a8", "w4a4"):
        body = sections[section]
        provenance = body["nll"]["provenance"]
        rules.append(
            CheckRule(
                path=f"/sections/{section}",
                comparator=ComparatorType.STRUCTURE,
                required_keys=sorted(body),
            )
        )
        for count_field in ("sample_count", "correct_count"):
            rules.append(
                CheckRule(
                    path=f"/sections/{section}/{count_field}",
                    comparator=ComparatorType.EXACT,
                    expected=body[count_field]["value"],
                    provenance=provenance,
                )
            )
        for metric in ("nll", "mean_margin"):
            rules.append(
                CheckRule(
                    path=f"/sections/{section}/{metric}",
                    comparator=ComparatorType.CLOSE,
                    expected=body[metric]["value"],
                    provenance=provenance,
                    **_TORCH_FLOAT,
                )
            )
        if "scales" in body:
            for site, leaf in sorted(body["scales"].items()):
                pointer = site.replace("~", "~0").replace("/", "~1")
                family = _NUMPY_SCALE if site == "__input__" else _TORCH_SCALE
                rules.append(
                    CheckRule(
                        path=f"/sections/{section}/scales/{pointer}",
                        comparator=ComparatorType.CLOSE,
                        expected=leaf["value"],
                        provenance="measured",
                        **family,
                    )
                )
            for site, leaf in sorted(body["zero_points"].items()):
                pointer = site.replace("~", "~0").replace("/", "~1")
                rules.append(
                    CheckRule(
                        path=f"/sections/{section}/zero_points/{pointer}",
                        comparator=ComparatorType.EXACT,
                        expected=leaf["value"],
                        provenance="measured",
                    )
                )
            input_elems = NUM_CALIB * 16 * 16
            for site, leaf in sorted(body["saturation"].items()):
                pointer = site.replace("~", "~0").replace("/", "~1")
                rules.append(
                    CheckRule(
                        path=f"/sections/{section}/saturation/{pointer}",
                        comparator=ComparatorType.CLOSE,
                        expected=leaf["value"],
                        provenance="simulated",
                        atol=_element_count(input_elems),
                        rtol=0.0,
                        rationale=(
                            "quotient of integer counts over "
                            f"{input_elems} fixed elements; any count change fails"
                        ),
                    )
                )

    spec = BaselineSpec(
        baseline_schema_version=BASELINE_SCHEMA_VERSION,
        baseline_name="smoke",
        description=(
            "Deterministic offline smoke artifact: seeded untrained TinyCNN, FP32 "
            "(measured) + simulated W8A8/W4A4 + representative qparams + input "
            "saturation + one estimated hardware cost. ADR-015 tolerances."
        ),
        artifact_type=ARTIFACT_TYPE,
        compatible_artifact_schema_versions=[ARTIFACT_SCHEMA_VERSION],
        capture_command=capture_command,
        environment_rules=EnvironmentRules(
            exact={"torch": "2.2.2"},
            allowed={"python_minor": ["3.11", "3.12"], "numpy": ["1.26.4"]},
            report_only=["platform", "python", "quantscope"],
        ),
        ignored_paths=["/environment/platform", "/environment/python"],
        rules=rules,
        canonical_digest="0" * 64,  # placeholder, replaced below
    )
    return spec.model_copy(update={"canonical_digest": spec.body_digest()})
