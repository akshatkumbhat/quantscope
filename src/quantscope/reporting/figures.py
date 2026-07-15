"""Deterministic report figures from run artifacts.

Rules (reporting-phase controls, ADR-012 addendum 6):

- The three checkpoint Pareto frontiers are NEVER averaged into one
  representative frontier; each checkpoint gets its own panel and the
  low cross-checkpoint overlap is printed on the figure.
- Provenance is labeled at the field level in each figure footer and
  in the returned manifest entry: fake-quant task metrics are
  *simulated*; analytical hardware cost is *estimated*; only real-INT8
  backend outputs from plan step C are *measured*.
- Per-checkpoint values are always shown; a mean never replaces them.
- Figures are byte-deterministic: fixed Agg backend, bundled DejaVu
  font, no timestamps in the output metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantscope.search import SweepRecord, pareto_frontier, pareto_jaccard

__all__ = [
    "fig_mechanism_decomposition",
    "fig_observer_factorial",
    "fig_pareto_frontiers",
    "fig_pow2_cost",
]

# Validated palette (dataviz reference instance, light mode).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
# Categorical slots 1-4, fixed order (worst adjacent CVD dE 24.2).
SERIES = {"minmax": "#2a78d6", "percentile": "#1baf7a", "mse_grid": "#eda100", "pow2": "#008300"}
# Ordinal blue ramp for the three decomposition stages (validated).
STAGE_RAMP = ("#86b6ef", "#2a78d6", "#104281")

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",  # bundled with matplotlib: deterministic
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "svg.hashsalt": "quantscope-report",
}
_PNG_META = {"Software": "quantscope build_report"}  # no timestamps


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata=_PNG_META)
    plt.close(fig)


def _footer(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=7.5, color=INK_SECONDARY)


def fig_pareto_frontiers(sweeps: dict[int, list[SweepRecord]], out_path: Path) -> dict[str, Any]:
    """B3 per-checkpoint Pareto panels (never averaged) + overlap text."""
    seeds = sorted(sweeps)
    frontiers = {s: pareto_frontier(sweeps[s], quality="nll") for s in seeds}
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(seeds), figsize=(3.4 * len(seeds), 3.2), sharey=True)
        axes = [axes] if len(seeds) == 1 else list(axes)
        for ax, seed in zip(axes, seeds, strict=True):
            records = sweeps[seed]
            frontier = sorted(frontiers[seed], key=lambda r: r.cost)
            dominated = [r for r in records if r not in frontier]
            ax.scatter(
                [r.cost for r in dominated],
                [r.nll for r in dominated],
                s=9,
                color=MUTED,
                alpha=0.35,
                linewidths=0,
                label="dominated",
            )
            ax.plot(
                [r.cost for r in frontier],
                [r.nll for r in frontier],
                marker="o",
                markersize=4.5,
                linewidth=2,
                color=SERIES["minmax"],
                label=f"Pareto frontier ({len(frontier)})",
            )
            ax.set_title(f"checkpoint seed {seed}", fontsize=9)
            ax.set_xlabel("analytical cost (estimated)")
        axes[0].set_ylabel("clean-eval NLL (simulated)")
        axes[0].legend(loc="upper right", fontsize=7.5)
        overlap = "  ·  ".join(
            f"J(s{a},s{b})={pareto_jaccard(frontiers[a], frontiers[b]):.3f}"
            for a in seeds
            for b in seeds
            if a < b
        )
        fig.suptitle(
            "B3 exhaustive mixed-precision sweeps: per-checkpoint Pareto frontiers (not averaged)",
            fontsize=10,
        )
        _footer(
            fig,
            "Frontier overlap (Jaccard on assignment sets): "
            f"{overlap} — frontiers barely overlap across checkpoints (ADR-010).\n"
            "Provenance: NLL simulated (fake-quant policy v1, not integer execution); "
            "cost estimated (normalized weight bits, analytical); nothing measured.",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save(fig, out_path)
    return {
        "figure": "pareto_frontiers",
        "provenance": {"nll": "simulated", "cost": "estimated"},
        "notes": "one panel per checkpoint; frontiers never averaged; Jaccard overlap on figure",
    }


def fig_observer_factorial(
    per_seed: dict[int, dict[str, dict[str, float]]],
    fp32_clean_nll: dict[int, float],
    out_path: Path,
) -> dict[str, Any]:
    """D primary result: W4A4 clean-eval NLL per observer, calibration
    condition, and checkpoint. ``per_seed[seed][observer]`` maps
    ``{"clean->clean": nll, "stressed->clean": nll}``.
    """
    seeds = sorted(per_seed)
    observers = list(SERIES)
    conditions = ("clean->clean", "stressed->clean")
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(seeds), figsize=(3.5 * len(seeds), 3.4), sharey=True)
        axes = [axes] if len(seeds) == 1 else list(axes)
        width = 0.38
        for ax, seed in zip(axes, seeds, strict=True):
            for i, obs in enumerate(observers):
                vals = [per_seed[seed][obs][c] for c in conditions]
                bars = ax.bar(
                    [i - width / 2, i + width / 2],
                    vals,
                    width - 0.04,
                    color=SERIES[obs],
                    edgecolor=SURFACE,
                    linewidth=2,
                )
                hatches = (None, "//")
                for bar, hatch, val in zip(bars, hatches, vals, strict=True):
                    if hatch:
                        bar.set_hatch(hatch)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color=INK_SECONDARY,
                        rotation=90 if val > 0.5 else 0,
                    )
            ax.axhline(
                fp32_clean_nll[seed], color=INK, linewidth=1, linestyle=":", label="FP32 (measured)"
            )
            ax.set_xticks(range(len(observers)))
            ax.set_xticklabels(observers, fontsize=8)
            ax.set_title(f"checkpoint seed {seed}", fontsize=9)
        axes[0].set_ylabel("clean-eval NLL (simulated)")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=MUTED),
            plt.Rectangle((0, 0), 1, 1, color=MUTED, hatch="//"),
            plt.Line2D([0], [0], color=INK, linewidth=1, linestyle=":"),
        ]
        axes[-1].legend(
            handles,
            ["clean calibration", "stressed calibration (7-sigma impulses)", "FP32 (measured)"],
            fontsize=7.5,
            loc="upper left",
        )
        fig.suptitle(
            "D observer study, W4A4 clean evaluation: per-checkpoint values (no averaging)",
            fontsize=10,
        )
        _footer(
            fig,
            "Provenance: quantized NLL simulated (fake-quant policy v1, not integer "
            "execution); FP32 reference measured.\nScope: input/early-activation "
            "calibration robustness under controlled impulse contamination — not "
            "network-wide observer superiority (ADR-012 addendum 6).",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _save(fig, out_path)
    return {
        "figure": "observer_factorial_w4a4",
        "provenance": {"quantized_nll": "simulated", "fp32_reference": "measured"},
        "notes": "per-checkpoint W4A4 NLL by observer and calibration condition",
    }


def fig_mechanism_decomposition(
    decomposition: dict[str, dict[str, Any]], out_path: Path
) -> dict[str, Any]:
    """Cumulative stage attribution of MinMax W4A4 damage per checkpoint."""
    seeds = sorted(decomposition)
    stages = ("input", "input+early", "input+early+deeper")
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        y = list(range(len(seeds)))
        height = 0.22
        for j, stage in enumerate(stages):
            vals = [decomposition[s]["cumulative_damage"][stage] for s in seeds]
            offs = [yy + (j - 1) * height for yy in y]
            ax.barh(
                offs,
                vals,
                height - 0.03,
                color=STAGE_RAMP[j],
                edgecolor=SURFACE,
                linewidth=1.5,
                label=f"substitute {stage}",
            )
            for yy, val in zip(offs, vals, strict=True):
                ax.text(
                    val,
                    yy,
                    f" {val:+.4f}",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color=INK_SECONDARY,
                )
        for i, s in enumerate(seeds):
            ax.plot(
                [decomposition[s]["total_damage"]] * 2,
                [i - 1.6 * height, i + 1.6 * height],
                color=INK,
                linewidth=1,
                linestyle=":",
            )
        ax.set_yticks(y)
        ax.set_yticklabels([str(s) for s in seeds])
        ax.set_ylabel("checkpoint")
        ax.invert_yaxis()
        ax.margins(x=0.14)  # room for the bar-end value labels
        ax.set_xlabel("Delta NLL vs clean calibration (simulated)")
        # Horizontal legend above the axes; inside placement collides
        # with the bar-end labels.
        ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=3)
        ax.set_title(
            "Mechanism decomposition: stressed MinMax qparams substituted cumulatively "
            "by stage (W4A4)",
            fontsize=10,
            pad=28,
        )
        _footer(
            fig,
            "Dotted line: total damage of fully stressed calibration. The input "
            "substitution alone reproduces >=95% of it on every checkpoint.\n"
            "Provenance: Delta NLL simulated (fake-quant policy v1, not integer "
            "execution).",
        )
        fig.tight_layout()
        _save(fig, out_path)
    return {
        "figure": "mechanism_decomposition",
        "provenance": {"delta_nll": "simulated"},
        "notes": "cumulative stage substitution, per checkpoint; input site dominates",
    }


def fig_pow2_cost(q3: dict[str, dict[str, dict[str, float]]], out_path: Path) -> dict[str, Any]:
    """Q3: power-of-two (round-up) NLL cost vs MinMax, per configuration,
    condition, and checkpoint. Panels have different y-scales (labeled)."""
    conditions = ("clean->clean", "stressed->clean")
    configs = list(q3)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
        for ax, cond in zip(axes, conditions, strict=True):
            for i, cfg in enumerate(configs):
                cells = q3[cfg][cond]
                seeds = sorted(cells)
                vals = [cells[k] for k in seeds]
                xs = [i + (j - 1) * 0.22 for j in range(len(vals))]
                ax.bar(
                    xs,
                    vals,
                    0.19,
                    color=SERIES["pow2"],
                    edgecolor=SURFACE,
                    linewidth=1.5,
                )
                for x, val in zip(xs, vals, strict=True):
                    ax.text(
                        x,
                        val,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=6.5,
                        color=INK_SECONDARY,
                        rotation=90,
                    )
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, fontsize=8)
            ax.axhline(0.0, color=BASELINE, linewidth=1)
            ax.margins(y=0.22)  # headroom so rotated value labels don't clip
            ax.set_title(f"{cond} (per-checkpoint bars)", fontsize=9, pad=10)
        axes[0].set_ylabel("NLL cost vs MinMax (simulated)")
        fig.suptitle(
            "Q3 (measurement only): power-of-two round-up scale cost — note the different y-scales",
            fontsize=10,
        )
        _footer(
            fig,
            "Applies specifically to the frozen round-up power-of-two policy under "
            "this study's contaminated-calibration condition.\nProvenance: NLL cost "
            "simulated (fake-quant policy v1, not integer execution).",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _save(fig, out_path)
    return {
        "figure": "pow2_cost",
        "provenance": {"nll_cost": "simulated"},
        "notes": "per-checkpoint values; panels intentionally use different y-scales",
    }
