"""Fast unit tests for per-group simulated quantization and ablation."""

import numpy as np
import pytest
import torch

from quantscope.data.texture10 import Texture10Params, make_texture10
from quantscope.models.bottleneck_resnet import BottleneckResNet
from quantscope.quantization.simulate import (
    BOTTLENECK_RESNET_GROUPS,
    SimQuantConfig,
    simulate_quantized_groups,
)
from quantscope.sensitivity import predictions

_SMALL = Texture10Params(num_classes=4, image_size=16)


def _model_and_data() -> tuple[BottleneckResNet, torch.utils.data.TensorDataset]:
    torch.manual_seed(0)
    model = BottleneckResNet(num_classes=4, bottleneck_width=4).eval()
    ds = make_texture10(num_samples=48, seed=1, params=_SMALL)
    return model, ds


class TestGroupPartition:
    def test_covers_every_weight_module_exactly_once(self) -> None:
        model = BottleneckResNet()
        weight_modules = {
            name
            for name, m in model.named_modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        }
        partitioned = [w for spec in BOTTLENECK_RESNET_GROUPS.values() for w in spec.weights]
        assert sorted(partitioned) == sorted(weight_modules)  # no gaps, no overlaps

    def test_covers_every_relu_exactly_once(self) -> None:
        model = BottleneckResNet()
        relus = {name for name, m in model.named_modules() if isinstance(m, torch.nn.ReLU)}
        partitioned = [a for spec in BOTTLENECK_RESNET_GROUPS.values() for a in spec.activations]
        assert sorted(partitioned) == sorted(relus)

    def test_exactly_one_group_owns_the_input(self) -> None:
        owners = [g for g, s in BOTTLENECK_RESNET_GROUPS.items() if s.include_input]
        assert owners == ["stem"]


class TestSimulateQuantizedGroups:
    def test_incomplete_assignment_rejected(self) -> None:
        model, ds = _model_and_data()
        with pytest.raises(ValueError, match="mismatch"):
            simulate_quantized_groups(model, ds, {"stem": SimQuantConfig(4, 4)})

    def test_all_fp32_is_identity(self) -> None:
        model, ds = _model_and_data()
        assignment = dict.fromkeys(BOTTLENECK_RESNET_GROUPS, None)
        sim = simulate_quantized_groups(model, ds, assignment)
        x = ds.tensors[0][:8]
        with torch.no_grad():
            assert torch.equal(model(x), sim(x))

    def test_only_targeted_group_weights_change(self) -> None:
        model, ds = _model_and_data()
        assignment: dict = dict.fromkeys(BOTTLENECK_RESNET_GROUPS, None)
        assignment["bottleneck"] = SimQuantConfig(4, 4)
        sim = simulate_quantized_groups(model, ds, assignment)
        assert not torch.equal(sim.bottleneck_conv.weight, model.bottleneck_conv.weight)
        assert torch.equal(sim.stem_conv.weight, model.stem_conv.weight)
        assert torch.equal(sim.classifier.weight, model.classifier.weight)

    def test_single_group_less_damage_than_uniform(self) -> None:
        model, ds = _model_and_data()
        x = ds.tensors[0][:16]
        one: dict = dict.fromkeys(BOTTLENECK_RESNET_GROUPS, None)
        one["down"] = SimQuantConfig(4, 4)
        all_groups = dict.fromkeys(BOTTLENECK_RESNET_GROUPS, SimQuantConfig(4, 4))
        sim_one = simulate_quantized_groups(model, ds, one)
        sim_all = simulate_quantized_groups(model, ds, all_groups)
        with torch.no_grad():
            ref = model(x)
            err_one = float((sim_one(x) - ref).pow(2).mean())
            err_all = float((sim_all(x) - ref).pow(2).mean())
        assert 0.0 < err_one < err_all

    def test_unknown_group_name_rejected(self) -> None:
        model, ds = _model_and_data()
        assignment = dict.fromkeys(BOTTLENECK_RESNET_GROUPS, None)
        assignment["nonexistent"] = SimQuantConfig(4, 4)
        with pytest.raises(ValueError, match="mismatch"):
            simulate_quantized_groups(model, ds, assignment)


class TestPredictions:
    def test_matches_dataset_order_and_size(self) -> None:
        model, ds = _model_and_data()
        preds = predictions(model, ds)
        assert preds.shape == (48,)
        assert preds.dtype == np.int64
        # Deterministic across calls.
        assert np.array_equal(preds, predictions(model, ds))
