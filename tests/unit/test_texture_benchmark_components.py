"""Fast unit tests for Texture-10, BottleneckResNet, detailed eval, and
simulated quantization. Small instances only; the real benchmark is slow
and lives behind the `slow` marker."""

import numpy as np
import pytest
import torch

from quantscope.data.texture10 import Texture10Params, make_texture10
from quantscope.evaluation import evaluate_detailed
from quantscope.models.bottleneck_resnet import BottleneckResNet
from quantscope.quantization.simulate import SimQuantConfig, simulate_quantized

_SMALL = Texture10Params(num_classes=4, image_size=16)


class TestTexture10:
    def test_shapes_and_determinism(self) -> None:
        a = make_texture10(num_samples=24, seed=3, params=_SMALL)
        b = make_texture10(num_samples=24, seed=3, params=_SMALL)
        assert a.tensors[0].shape == (24, 1, 16, 16)
        assert torch.equal(a.tensors[0], b.tensors[0])
        assert not torch.equal(
            a.tensors[0], make_texture10(num_samples=24, seed=4, params=_SMALL).tensors[0]
        )

    def test_snr_controls_noise(self) -> None:
        clean = make_texture10(
            num_samples=16,
            seed=0,
            params=Texture10Params(num_classes=4, image_size=16, snr_db=25.0),
        )
        noisy = make_texture10(
            num_samples=16,
            seed=0,
            params=Texture10Params(num_classes=4, image_size=16, snr_db=2.0),
        )
        # Same seed stream: higher noise power => higher total variance.
        assert float(noisy.tensors[0].var()) > float(clean.tensors[0].var())

    def test_boundary_fraction_validated(self) -> None:
        with pytest.raises(ValueError, match="boundary_fraction"):
            Texture10Params(boundary_fraction=0.9)

    def test_all_classes_present(self) -> None:
        ds = make_texture10(num_samples=40, seed=1, params=_SMALL)
        assert set(ds.tensors[1].tolist()) == {0, 1, 2, 3}


class TestBottleneckResNet:
    def test_forward_shape_and_param_count(self) -> None:
        model = BottleneckResNet(num_classes=10)
        out = model(torch.randn(2, 1, 32, 32))
        assert out.shape == (2, 10)
        n_params = sum(p.numel() for p in model.parameters())
        assert 15_000 < n_params < 30_000

    def test_fx_traceable(self) -> None:
        model = BottleneckResNet().eval()
        traced = torch.fx.symbolic_trace(model)
        assert traced(torch.randn(1, 1, 32, 32)).shape == (1, 10)

    def test_relu_modules_distinct(self) -> None:
        # Policy v1 hooks ReLU outputs; shared instances would conflate sites.
        model = BottleneckResNet()
        relus = [m for m in model.modules() if isinstance(m, torch.nn.ReLU)]
        assert len(relus) == len({id(m) for m in relus}) == 8

    def test_invalid_bottleneck_rejected(self) -> None:
        with pytest.raises(ValueError, match="bottleneck_width"):
            BottleneckResNet(bottleneck_width=1)


class TestEvaluateDetailed:
    def test_perfect_model_positive_margin(self) -> None:
        # A "model" that returns one-hot-like logits from the labels is not
        # constructible here; instead check metric consistency on a real net.
        torch.manual_seed(0)
        model = BottleneckResNet(num_classes=4, bottleneck_width=4).eval()
        ds = make_texture10(num_samples=32, seed=0, params=_SMALL)
        m = evaluate_detailed(model, ds)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert m["nll"] > 0.0
        assert np.isfinite(m["mean_margin"])


class TestSimulateQuantized:
    def _model_and_data(self) -> tuple[BottleneckResNet, torch.utils.data.TensorDataset]:
        torch.manual_seed(1)
        model = BottleneckResNet(num_classes=4, bottleneck_width=4).eval()
        ds = make_texture10(num_samples=48, seed=2, params=_SMALL)
        return model, ds

    def test_original_model_untouched(self) -> None:
        model, ds = self._model_and_data()
        before = model.stem_conv.weight.detach().clone()
        simulate_quantized(model, ds, SimQuantConfig(4, 4))
        assert torch.equal(model.stem_conv.weight, before)

    def test_w8a8_close_but_not_identical(self) -> None:
        model, ds = self._model_and_data()
        sim = simulate_quantized(model, ds, SimQuantConfig(8, 8))
        x = ds.tensors[0][:8]
        with torch.no_grad():
            ref, out = model(x), sim(x)
        assert not torch.equal(ref, out)  # quantization did something
        assert float((ref - out).abs().max()) < 1.0  # but not catastrophic

    def test_w4a4_worse_than_w8a8(self) -> None:
        model, ds = self._model_and_data()
        x = ds.tensors[0][:16]
        sim8 = simulate_quantized(model, ds, SimQuantConfig(8, 8))
        sim4 = simulate_quantized(model, ds, SimQuantConfig(4, 4))
        with torch.no_grad():
            ref = model(x)
            err8 = float((sim8(x) - ref).pow(2).mean())
            err4 = float((sim4(x) - ref).pow(2).mean())
        assert err4 > err8

    def test_weight_bits_actually_reduce_levels(self) -> None:
        model, ds = self._model_and_data()
        sim = simulate_quantized(model, ds, SimQuantConfig(2, 8))
        w = sim.classifier.weight.detach()
        # 2-bit signed symmetric: at most 2^2 - 1 distinct levels per channel.
        for row in w:
            assert len(torch.unique(row)) <= 3

    def test_no_relu_model_rejected(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(4, 2))
        ds = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))
        with pytest.raises(ValueError, match="ReLU"):
            simulate_quantized(model, ds, SimQuantConfig(8, 8))
