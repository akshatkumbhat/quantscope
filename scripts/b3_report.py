#!/usr/bin/env python
"""B3 report: within-checkpoint search utility and cross-checkpoint transfer.

Executes exactly the ADR-010 predeclared analysis over the three sweep
tables and the three W4A4 ablation artifacts. Reports the two questions
separately and never pools them:

  1. Within-checkpoint utility — sensitivity-guided and greedy search vs
     random search and the exact optimum, per checkpoint.
  2. Cross-checkpoint transfer — seed A's ranking walked on seed B's
     table, all 6 ordered pairs, vs B's own exact optimum.

Provenance: NLL/accuracy simulated; cost estimated (normalized weight
bits). Exit code 0 if the predeclared B3 success criteria hold.
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

from quantscope.quantization.simulate import BOTTLENECK_RESNET_GROUPS
from quantscope.search import (
    SweepRecord,
    budget_regret,
    evals_to_frontier,
    greedy_path,
    pareto_frontier,
    pareto_jaccard,
    random_search_regrets,
    sensitivity_path,
)
from quantscope.search.analysis import BUDGET, DELTA_NLL

SEEDS = (0, 1, 2)
GROUP_NAMES = list(BOTTLENECK_RESNET_GROUPS)


def load_sweep(base: Path, seed: int) -> list[SweepRecord]:
    path = base / f"texture-a-seed{seed}-sweep" / "sweep_table.json"
    rows = json.loads(path.read_text())
    return [
        SweepRecord(bits=tuple(r["bits"]), cost=r["cost"], nll=r["nll"], accuracy=r["accuracy"])
        for r in rows
    ]


def load_ranking(base: Path, seed: int) -> list[str]:
    """Groups ordered least-sensitive first (ascending W4A4 ΔNLL)."""
    path = base / f"texture-a-seed{seed}-ablation-w4a4" / "metrics.json"
    metrics = json.loads(path.read_text())["metrics"]
    dnll: dict[str, float] = {}
    for m in metrics:
        if m["name"].endswith("_delta_nll"):
            dnll[m["name"][: -len("_delta_nll")]] = m["value"]
    if sorted(dnll) != sorted(GROUP_NAMES):
        raise ValueError(f"ablation groups mismatch for seed {seed}")
    return sorted(GROUP_NAMES, key=lambda g: dnll[g])


def main(base: str = "runs/validation-012") -> int:
    base_path = Path(base)
    tables = {s: load_sweep(base_path, s) for s in SEEDS}
    rankings = {s: load_ranking(base_path, s) for s in SEEDS}

    print(f"predeclared: budget={BUDGET}, delta={DELTA_NLL} NLL, random=10x32\n")
    failures: list[str] = []

    # ---- Question 1: within-checkpoint utility --------------------------
    print("== Q1: within-checkpoint utility ==")
    header = (
        f"{'seed':<6}{'exactNLL':<10}{'sens regret':<13}{'sens evals':<12}"
        f"{'greedy regret':<15}{'greedy evals':<14}{'random median':<15}{'random range'}"
    )
    print(header)
    for s in SEEDS:
        records = tables[s]
        sens = sensitivity_path(rankings[s], GROUP_NAMES, records)
        greedy = greedy_path(GROUP_NAMES, records)
        sens_regret = budget_regret(sens, records)["nll_regret"]
        greedy_regret = budget_regret(greedy, records)["nll_regret"]
        sens_evals = evals_to_frontier(sens, records)
        greedy_evals = evals_to_frontier(greedy, records)
        randoms = random_search_regrets(records)
        exact = min(r.nll for r in records if r.cost <= BUDGET + 1e-9)
        print(
            f"{s:<6}{exact:<10.4f}{sens_regret:<13.4f}{sens_evals!s:<12}"
            f"{greedy_regret:<15.4f}{greedy_evals!s:<14}"
            f"{statistics.median(randoms):<15.4f}"
            f"[{min(randoms):.4f}, {max(randoms):.4f}]"
        )

    # Nontrivial tradeoff: spread of NLL among budget-feasible configs and
    # a frontier with intermediate (mixed) points.
    print("\n== tradeoff nontriviality ==")
    for s in SEEDS:
        feasible = [r for r in tables[s] if r.cost <= BUDGET + 1e-9]
        spread = max(r.nll for r in feasible) - min(r.nll for r in feasible)
        frontier = pareto_frontier(tables[s], quality="nll")
        mixed = [r for r in frontier if len(set(r.bits)) > 1]
        print(
            f"seed {s}: NLL spread within budget {spread:.4f}; "
            f"frontier size {len(frontier)} ({len(mixed)} mixed-precision points)"
        )
        if spread < 2 * DELTA_NLL:
            failures.append(f"seed {s}: budget-feasible NLL spread {spread:.4f} < {2 * DELTA_NLL}")
        if not mixed:
            failures.append(f"seed {s}: frontier contains no mixed-precision points")

    # ---- Question 2: cross-checkpoint transfer --------------------------
    print("\n== Q2: cross-checkpoint transfer (ranking from A applied to B) ==")
    print(f"{'A->B':<8}{'transfer regret':<17}{'B own-rank regret':<19}{'penalty'}")
    for a, b in itertools.permutations(SEEDS, 2):
        transferred = sensitivity_path(rankings[a], GROUP_NAMES, tables[b])
        own = sensitivity_path(rankings[b], GROUP_NAMES, tables[b])
        t_regret = budget_regret(transferred, tables[b])["nll_regret"]
        o_regret = budget_regret(own, tables[b])["nll_regret"]
        print(f"{a}->{b:<6}{t_regret:<17.4f}{o_regret:<19.4f}{t_regret - o_regret:+.4f}")

    print("\n== Pareto overlap across checkpoints (Jaccard, NLL frontier) ==")
    for a, b in itertools.combinations(SEEDS, 2):
        fa = pareto_frontier(tables[a], quality="nll")
        fb = pareto_frontier(tables[b], quality="nll")
        print(f"seeds {a}-{b}: {pareto_jaccard(fa, fb):.3f}")

    if failures:
        print("\nB3 SUCCESS CRITERIA NOT MET:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nB3 success criteria met: nontrivial tradeoff; searches evaluated vs exact optima.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
