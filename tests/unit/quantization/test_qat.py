"""Unit tests for the ADR-013 Torch fake-quant adapter and QAT loop.

Covers: NumPy/Torch forward parity, clipped-STE gradient policy,
clipping/saturation boundaries, frozen activation qparams, weight-rule
behavior, deterministic fine-tuning on a tiny fixture, and actionable
failure on incompatible or missing qparams.
"""

import numpy as np
import pytest
import torch

from quantscope.data.texture10 import Texture10Params, make_texture10
from quantscope.models.bottleneck_resnet import BottleneckResNet
from quantscope.quantization.affine import (
    Granularity,
    QuantParams,
    Scheme,
    compute_quant_params,
    fake_quantize,
)
from quantscope.quantization.qat import QATRecipe, qat_finetune, torch_fake_quantize
from quantscope.quantization.simulate import calibrate_activation_params

_SMALL = Texture10Params(num_classes=4, image_size=16)


def _model() -> BottleneckResNet:
    torch.manual_seed(0)
    return BottleneckResNet(num_classes=4, bottleneck_width=4).eval()


def _tiny_data(n: int = 48, seed: int = 0):
    return make_texture10(num_samples=n, seed=seed, params=_SMALL)


class TestForwardParity:
    """The Torch adapter must reproduce the NumPy reference exactly."""

    @pytest.mark.parametrize("bits", [4, 8])
    def test_per_tensor_asymmetric_unsigned(self, bits: int) -> None:
        rng = np.random.default_rng(1)
        values = np.abs(rng.normal(size=(6, 5, 4, 4))).astype(np.float32)
        params = compute_quant_params(values, bits=bits, signed=False, scheme=Scheme.ASYMMETRIC)
        expected = fake_quantize(values, params)
        actual = torch_fake_quantize(torch.from_numpy(values), params)
        np.testing.assert_array_equal(actual.numpy(), expected)

    @pytest.mark.parametrize("bits", [4, 8])
    def test_per_tensor_asymmetric_signed_input(self, bits: int) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(size=(8, 1, 16, 16)).astype(np.float32)
        params = compute_quant_params(values, bits=bits, signed=True, scheme=Scheme.ASYMMETRIC)
        expected = fake_quantize(values, params)
        actual = torch_fake_quantize(torch.from_numpy(values), params)
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_per_channel_symmetric_weights(self) -> None:
        rng = np.random.default_rng(3)
        weight = rng.normal(size=(7, 3, 3, 3)).astype(np.float32)
        params = compute_quant_params(
            weight,
            bits=4,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        expected = fake_quantize(weight, params)
        actual = torch_fake_quantize(torch.from_numpy(weight), params)
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_weight_rule_matches_simulate_quantized_weights(self) -> None:
        # The per-forward weight rule inside qat_finetune must equal the
        # reference compute-then-fake-quantize on identical weights.
        from quantscope.quantization.qat import _WeightFakeQuant

        rng = np.random.default_rng(4)
        weight = rng.normal(size=(5, 4, 3, 3)).astype(np.float32)
        params = compute_quant_params(
            weight,
            bits=4,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        expected = fake_quantize(weight, params)
        actual = _WeightFakeQuant(bits=4)(torch.from_numpy(weight))
        np.testing.assert_array_equal(actual.detach().numpy(), expected)

    def test_degenerate_all_zero_channel(self) -> None:
        weight = np.zeros((3, 2, 2, 2), dtype=np.float32)
        weight[0] = 0.5  # one live channel; two degenerate
        params = compute_quant_params(
            weight,
            bits=4,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        expected = fake_quantize(weight, params)
        actual = torch_fake_quantize(torch.from_numpy(weight), params)
        np.testing.assert_array_equal(actual.numpy(), expected)


def _scalar_params(scale: float, zero_point: int, bits: int = 4, *, signed: bool = True):
    from quantscope.quantization.affine import integer_range

    r = integer_range(bits, signed=signed)
    return QuantParams(
        scale=np.float32(scale),
        zero_point=np.int32(zero_point),
        qmin=r.qmin,
        qmax=r.qmax,
        bits=bits,
        signed=signed,
        scheme=Scheme.ASYMMETRIC,
        granularity=Granularity.PER_TENSOR,
        channel_axis=None,
    )


class TestClippedSTE:
    def test_gradient_passes_in_range_zero_when_saturated(self) -> None:
        # scale 1, zp 0, 4-bit signed: representable codes [-8, 7].
        params = _scalar_params(1.0, 0)
        x = torch.tensor([-9.0, -8.0, 0.3, 6.9, 7.0, 7.6, 42.0], requires_grad=True)
        y = torch_fake_quantize(x, params)
        y.sum().backward()
        # 7.6 rounds to 8 > qmax: saturated. 7.4 would round to 7: passes.
        np.testing.assert_array_equal(x.grad.numpy(), [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    def test_saturation_boundary_uses_preclamp_code(self) -> None:
        params = _scalar_params(1.0, 0)
        # 7.5 rounds half-to-even to 8 (saturated); 6.5 rounds to 6 (in range).
        x = torch.tensor([6.5, 7.5], requires_grad=True)
        y = torch_fake_quantize(x, params)
        y.sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), [1.0, 0.0])
        np.testing.assert_array_equal(y.detach().numpy(), [6.0, 7.0])  # forward saturates

    def test_gradients_finite_for_nonsaturated_values(self) -> None:
        params = _scalar_params(0.05, 3)
        x = torch.linspace(-0.2, 0.2, 41, requires_grad=True)
        torch_fake_quantize(x, params).pow(2).sum().backward()
        assert torch.all(torch.isfinite(x.grad))

    def test_same_policy_for_weights(self) -> None:
        from quantscope.quantization.qat import _WeightFakeQuant

        w = torch.tensor([[1.0, -2.0, 4.0]], requires_grad=True)  # scale = 4/7
        out = _WeightFakeQuant(bits=4)(w)
        out.sum().backward()
        # bound/qmax scale keeps every entry's |code| <= 7: all pass.
        np.testing.assert_array_equal(w.grad.numpy(), [[1.0, 1.0, 1.0]])
        assert w.grad is not None  # scales detached, weight still trained


class TestFrozenQparams:
    def test_activation_qparams_unchanged_by_training_step(self) -> None:
        model = _model()
        data = _tiny_data()
        act_params = calibrate_activation_params(model, data, bits=4)
        before = {k: (np.copy(v.scale), np.copy(v.zero_point)) for k, v in act_params.items()}
        recipe = QATRecipe(learning_rate=1e-3, epochs=1, batch_size=16, seed=0)
        qat_finetune(model, data, act_params, recipe)
        for site, params in act_params.items():
            np.testing.assert_array_equal(params.scale, before[site][0])
            np.testing.assert_array_equal(params.zero_point, before[site][1])

    def test_finetune_changes_weights_but_not_original_model(self) -> None:
        model = _model()
        data = _tiny_data()
        act_params = calibrate_activation_params(model, data, bits=4)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        tuned, history = qat_finetune(
            model, data, act_params, QATRecipe(learning_rate=1e-3, epochs=1, batch_size=16)
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])
        moved = any(
            not torch.equal(tuned.state_dict()[k], original[k])
            for k in original
            if original[k].dtype.is_floating_point
        )
        assert moved, "fine-tuning did not change any weight"
        assert history.gradients_finite
        assert len(history.epoch_train_loss) == 1
        # Parametrizations removed: plain FP32 weights on the result.
        assert not any("parametrizations" in k for k in tuned.state_dict())


class TestDeterminism:
    def test_two_runs_identical(self) -> None:
        data = _tiny_data()
        results = []
        for _ in range(2):
            model = _model()  # re-seeded identically inside _model()
            act_params = calibrate_activation_params(model, data, bits=4)
            tuned, history = qat_finetune(
                model, data, act_params, QATRecipe(learning_rate=3e-4, epochs=2, batch_size=16)
            )
            results.append((tuned.state_dict(), history.epoch_train_loss))
        for key in results[0][0]:
            torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0.0, atol=0.0)
        assert results[0][1] == results[1][1]


class TestFailureModes:
    def test_missing_activation_site_rejected(self) -> None:
        model = _model()
        data = _tiny_data()
        act_params = calibrate_activation_params(model, data, bits=4)
        incomplete = {k: v for k, v in act_params.items() if k != "stem_relu"}
        with pytest.raises(ValueError, match="stem_relu"):
            qat_finetune(model, data, incomplete, QATRecipe(learning_rate=1e-3, epochs=1))

    def test_per_channel_activation_params_rejected(self) -> None:
        model = _model()
        data = _tiny_data()
        act_params = dict(calibrate_activation_params(model, data, bits=4))
        rng = np.random.default_rng(0)
        bad = compute_quant_params(
            rng.normal(size=(4, 3)).astype(np.float32),
            bits=4,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        act_params["stem_relu"] = bad
        with pytest.raises(ValueError, match="per-tensor"):
            qat_finetune(model, data, act_params, QATRecipe(learning_rate=1e-3, epochs=1))

    def test_channel_mismatch_rejected(self) -> None:
        rng = np.random.default_rng(0)
        weight = rng.normal(size=(5, 2, 3, 3)).astype(np.float32)
        params = compute_quant_params(
            weight,
            bits=4,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        with pytest.raises(ValueError, match="channels"):
            torch_fake_quantize(torch.zeros(3, 2, 3, 3), params)
