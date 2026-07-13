"""Search-method evaluation against an exhaustive sweep table (ADR-010).

All searches are *simulated on the table*: the exhaustive sweep already
evaluated every configuration, so search strategies become lookups and
their regret against the exact optimum is computed, not estimated.

Two separately-reported questions:
- within-checkpoint utility: does sensitivity-guided search beat random
  search on the checkpoint its ranking came from?
- cross-checkpoint transfer: does a ranking from one checkpoint remain
  useful on another? (Measured, never assumed.)
"""

from __future__ import annotations

import numpy as np

from quantscope.search.exhaustive import SweepRecord

__all__ = [
    "budget_regret",
    "evals_to_frontier",
    "greedy_path",
    "pareto_jaccard",
    "random_search_regrets",
    "sensitivity_path",
]

BUDGET = 0.75  # predeclared fixed cost budget (ADR-010)
DELTA_NLL = 0.01  # predeclared frontier-distance threshold


def _table(records: list[SweepRecord]) -> dict[tuple[int, ...], SweepRecord]:
    return {r.bits: r for r in records}


def _exact_best(records: list[SweepRecord], budget: float) -> SweepRecord:
    feasible = [r for r in records if r.cost <= budget + 1e-9]
    if not feasible:
        raise ValueError(f"no configuration within budget {budget}")
    return min(feasible, key=lambda r: r.nll)


def budget_regret(
    visited: list[SweepRecord], records: list[SweepRecord], *, budget: float = BUDGET
) -> dict[str, float]:
    """NLL/accuracy regret of the best *visited* config within budget vs the
    exact optimum within budget."""
    exact = _exact_best(records, budget)
    feasible = [r for r in visited if r.cost <= budget + 1e-9]
    if not feasible:
        return {"nll_regret": float("inf"), "accuracy_regret": float("inf")}
    found = min(feasible, key=lambda r: r.nll)
    return {
        "nll_regret": found.nll - exact.nll,
        "accuracy_regret": exact.accuracy - found.accuracy,
    }


def evals_to_frontier(
    visited: list[SweepRecord],
    records: list[SweepRecord],
    *,
    budget: float = BUDGET,
    delta: float = DELTA_NLL,
) -> int | None:
    """Number of evaluations until budget-regret <= delta; None if never."""
    for i in range(1, len(visited) + 1):
        if budget_regret(visited[:i], records, budget=budget)["nll_regret"] <= delta:
            return i
    return None


def sensitivity_path(
    ranking: list[str],
    group_names: list[str],
    records: list[SweepRecord],
) -> list[SweepRecord]:
    """Flip groups to INT4 in the given order (least sensitive first),
    starting from all-INT8: 9 path points, looked up in the table."""
    if sorted(ranking) != sorted(group_names):
        raise ValueError("ranking must be a permutation of the group names")
    table = _table(records)
    bits = dict.fromkeys(group_names, 8)
    path = [table[tuple(bits[g] for g in group_names)]]
    for group in ranking:
        bits[group] = 4
        path.append(table[tuple(bits[g] for g in group_names)])
    return path


def greedy_path(group_names: list[str], records: list[SweepRecord]) -> list[SweepRecord]:
    """From all-INT8, commit the lowest-NLL flip each round (36 evals).

    Returns the visited configs in evaluation order (candidate evals
    included, since regret-vs-evals must count every table lookup)."""
    table = _table(records)
    bits = dict.fromkeys(group_names, 8)
    visited: list[SweepRecord] = [table[tuple(bits[g] for g in group_names)]]
    remaining = list(group_names)
    while remaining:
        candidates = []
        for group in remaining:
            trial = dict(bits, **{group: 4})
            record = table[tuple(trial[g] for g in group_names)]
            visited.append(record)
            candidates.append((record.nll, group))
        _, best_group = min(candidates)
        bits[best_group] = 4
        remaining.remove(best_group)
    return visited


def random_search_regrets(
    records: list[SweepRecord],
    *,
    num_seeds: int = 10,
    num_evals: int = 32,
    budget: float = BUDGET,
) -> list[float]:
    """Best budget-regret of uniform random search, per deterministic seed."""
    regrets = []
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(records), size=num_evals, replace=False)
        visited = [records[i] for i in idx]
        regrets.append(budget_regret(visited, records, budget=budget)["nll_regret"])
    return regrets


def pareto_jaccard(frontier_a: list[SweepRecord], frontier_b: list[SweepRecord]) -> float:
    """Jaccard similarity of two frontiers' assignment sets."""
    a = {r.bits for r in frontier_a}
    b = {r.bits for r in frontier_b}
    return len(a & b) / len(a | b)
