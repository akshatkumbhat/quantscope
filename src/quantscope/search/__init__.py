"""Mixed-precision search: exhaustive sweeps and search-method analysis."""

from quantscope.search.analysis import (
    budget_regret,
    evals_to_frontier,
    greedy_path,
    pareto_jaccard,
    random_search_regrets,
    sensitivity_path,
)
from quantscope.search.exhaustive import (
    SweepRecord,
    config_cost,
    exhaustive_sweep,
    group_param_counts,
    pareto_frontier,
)

__all__ = [
    "SweepRecord",
    "budget_regret",
    "config_cost",
    "evals_to_frontier",
    "exhaustive_sweep",
    "greedy_path",
    "group_param_counts",
    "pareto_frontier",
    "pareto_jaccard",
    "random_search_regrets",
    "sensitivity_path",
]
