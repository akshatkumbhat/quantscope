"""Unit tests for exhaustive sweep machinery and table-based search analysis."""

import pytest
import torch

from quantscope.data.texture10 import Texture10Params, make_texture10
from quantscope.models.bottleneck_resnet import BottleneckResNet
from quantscope.quantization.simulate import GroupSpec
from quantscope.search import (
    SweepRecord,
    budget_regret,
    config_cost,
    evals_to_frontier,
    exhaustive_sweep,
    greedy_path,
    group_param_counts,
    pareto_frontier,
    pareto_jaccard,
    random_search_regrets,
    sensitivity_path,
)


def _toy_records() -> list[SweepRecord]:
    """Four configs over 2 groups with a known frontier."""
    return [
        SweepRecord(bits=(8, 8), cost=1.00, nll=0.10, accuracy=0.96),
        SweepRecord(bits=(4, 8), cost=0.80, nll=0.12, accuracy=0.95),
        SweepRecord(bits=(8, 4), cost=0.70, nll=0.30, accuracy=0.90),
        SweepRecord(bits=(4, 4), cost=0.50, nll=0.35, accuracy=0.89),
    ]


class TestCostModel:
    def test_uniform_bounds(self) -> None:
        counts = {"a": 100, "b": 300}
        assert config_cost((8, 8), counts) == pytest.approx(1.0)
        assert config_cost((4, 4), counts) == pytest.approx(0.5)

    def test_weighted_by_params(self) -> None:
        counts = {"a": 100, "b": 300}
        # Quantizing the larger group saves more.
        assert config_cost((8, 4), counts) < config_cost((4, 8), counts)

    def test_wrong_arity_rejected(self) -> None:
        with pytest.raises(ValueError, match="bit choices"):
            config_cost((8,), {"a": 1, "b": 1})

    def test_group_param_counts_cover_model(self) -> None:
        model = BottleneckResNet()
        counts = group_param_counts(model)
        expected = sum(
            m.weight.numel()
            for m in model.modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        )
        assert sum(counts.values()) == expected


class TestParetoFrontier:
    def test_nll_frontier(self) -> None:
        frontier = pareto_frontier(_toy_records(), quality="nll")
        assert [r.bits for r in frontier] == [(4, 4), (8, 4), (4, 8), (8, 8)]

    def test_dominated_config_excluded(self) -> None:
        records = [
            *_toy_records(),
            SweepRecord(bits=(9, 9), cost=0.80, nll=0.40, accuracy=0.80),  # dominated
        ]
        frontier = pareto_frontier(records, quality="nll")
        assert (9, 9) not in {r.bits for r in frontier}

    def test_equal_cost_keeps_best_only(self) -> None:
        records = [
            SweepRecord(bits=(1,), cost=0.5, nll=0.2, accuracy=0.9),
            SweepRecord(bits=(2,), cost=0.5, nll=0.1, accuracy=0.95),
        ]
        for quality in ("nll", "accuracy"):
            frontier = pareto_frontier(records, quality=quality)
            assert [r.bits for r in frontier] == [(2,)]

    def test_unknown_quality_rejected(self) -> None:
        with pytest.raises(ValueError, match="quality"):
            pareto_frontier(_toy_records(), quality="vibes")


class TestSearchAnalysis:
    def test_budget_regret_exact_when_optimum_visited(self) -> None:
        records = _toy_records()
        regret = budget_regret(records, records, budget=0.8)
        assert regret["nll_regret"] == pytest.approx(0.0)

    def test_budget_regret_positive_when_missed(self) -> None:
        records = _toy_records()
        visited = [records[2]]  # (8,4): nll 0.30; exact best <=0.8 is (4,8): 0.12
        regret = budget_regret(visited, records, budget=0.8)
        assert regret["nll_regret"] == pytest.approx(0.18)

    def test_budget_regret_inf_when_nothing_feasible(self) -> None:
        records = _toy_records()
        assert budget_regret([records[0]], records, budget=0.8)["nll_regret"] == float("inf")

    def test_evals_to_frontier(self) -> None:
        records = _toy_records()
        visited = [records[3], records[2], records[1]]  # hits optimum on 3rd
        assert evals_to_frontier(visited, records, budget=0.8, delta=0.01) == 3

    def test_sensitivity_path_walks_table(self) -> None:
        records = _toy_records()
        path = sensitivity_path(["b", "a"], ["a", "b"], records)
        assert [r.bits for r in path] == [(8, 8), (8, 4), (4, 4)]

    def test_sensitivity_path_validates_ranking(self) -> None:
        with pytest.raises(ValueError, match="permutation"):
            sensitivity_path(["a", "a"], ["a", "b"], _toy_records())

    def test_greedy_path_eval_count_and_choice(self) -> None:
        records = _toy_records()
        visited = greedy_path(["a", "b"], records)
        # 1 start + 2 candidates + 1 candidate = 4 evaluations for 2 groups.
        assert len(visited) == 4
        # First committed flip must be group "a" ((4,8): nll 0.12 < 0.30).
        assert visited[-1].bits == (4, 4)

    def test_random_search_deterministic(self) -> None:
        records = _toy_records()
        a = random_search_regrets(records, num_seeds=3, num_evals=2, budget=0.8)
        b = random_search_regrets(records, num_seeds=3, num_evals=2, budget=0.8)
        assert a == b
        assert len(a) == 3

    def test_pareto_jaccard(self) -> None:
        fa = pareto_frontier(_toy_records(), quality="nll")
        assert pareto_jaccard(fa, fa) == 1.0
        assert pareto_jaccard(fa, fa[:2]) == pytest.approx(0.5)


class TestExhaustiveSweepSmall:
    def test_two_group_sweep_on_tiny_model(self) -> None:
        # Full machinery, reduced to 2 groups => 4 configs: fast enough for CI.
        torch.manual_seed(0)
        model = BottleneckResNet(num_classes=4, bottleneck_width=4).eval()
        ds = make_texture10(
            num_samples=32, seed=0, params=Texture10Params(num_classes=4, image_size=16)
        )
        groups = {
            "stem": GroupSpec(("stem_conv",), ("stem_relu",), include_input=True),
            "classifier": GroupSpec(("classifier",), ()),
        }
        records = exhaustive_sweep(model, ds, ds, groups=groups, batch_size=16)
        assert len(records) == 4
        assert {r.bits for r in records} == {(4, 4), (4, 8), (8, 4), (8, 8)}
        costs = {r.bits: r.cost for r in records}
        assert costs[(8, 8)] == pytest.approx(1.0)
        assert costs[(4, 4)] == pytest.approx(0.5)
