"""C1 tests: torch_2_2 qparam policy, contained observer extraction, and
primitive fake-quant parity against Torch's own ops (ADR-011)."""

import typing

import numpy as np
import pytest
import torch
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver

from quantscope.quantization.affine import (
    Granularity,
    Scheme,
    compute_quant_params,
    dequantize,
    fake_quantize,
    quantize,
)
from quantscope.quantization.backend_matched import (
    FrozenQuantScopeFakeQuant,
    extract_histogram_bounds,
    extract_weight_bounds,
    quantscope_params_from_bounds,
)


class TestTorch22Policy:
    def test_symmetric_scale_denominator(self) -> None:
        values = np.array([-2.0, 1.0], dtype=np.float32)
        ours = compute_quant_params(values, bits=8, signed=True, scheme=Scheme.SYMMETRIC)
        compat = compute_quant_params(
            values, bits=8, signed=True, scheme=Scheme.SYMMETRIC, qparam_policy="torch_2_2"
        )
        assert float(ours.scale) == pytest.approx(2.0 / 127, rel=1e-6)
        assert float(compat.scale) == pytest.approx(2.0 / 127.5, rel=1e-6)
        # The documented ~0.39% systematic difference.
        assert float(ours.scale) / float(compat.scale) == pytest.approx(127.5 / 127, rel=1e-6)

    def test_asymmetric_unaffected_by_policy(self) -> None:
        values = np.array([-1.0, 3.0], dtype=np.float32)
        a = compute_quant_params(values, scheme=Scheme.ASYMMETRIC)
        b = compute_quant_params(values, scheme=Scheme.ASYMMETRIC, qparam_policy="torch_2_2")
        assert float(a.scale) == float(b.scale)
        assert int(a.zero_point) == int(b.zero_point)

    def test_unknown_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="policy"):
            compute_quant_params(np.array([1.0]), qparam_policy="torch_9_9")

    def test_per_channel_matches_torch_weight_observer(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(6, 3, 3, 3)
        observer = PerChannelMinMaxObserver(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0
        )
        observer(weight)
        torch_scale, torch_zp = observer.calculate_qparams()

        stacked = np.stack([observer.min_val.numpy(), observer.max_val.numpy()]).astype(np.float32)
        ours = compute_quant_params(
            stacked,
            bits=8,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=-1,
            qparam_policy="torch_2_2",
        )
        np.testing.assert_allclose(ours.scale, torch_scale.numpy(), rtol=1e-5)
        np.testing.assert_array_equal(ours.zero_point, torch_zp.numpy())


class TestHistogramExtraction:
    def _calibrated_observer(self) -> HistogramObserver:
        torch.manual_seed(1)
        obs = HistogramObserver(dtype=torch.quint8, reduce_range=True)
        for _ in range(4):
            obs(torch.rand(512) * 3.0)
        return obs

    def test_characterization_bounds_reproduce_calculate_qparams(self) -> None:
        # ADR-011: qparams computed from the extracted searched bounds must
        # equal observer.calculate_qparams() under Torch semantics.
        obs = self._calibrated_observer()
        torch_scale, torch_zp = obs.calculate_qparams()
        bounds = extract_histogram_bounds(obs)
        ours = quantscope_params_from_bounds(bounds, qparam_policy="torch_2_2")
        assert float(ours.scale) == pytest.approx(float(torch_scale), rel=1e-5)
        assert int(ours.zero_point) == int(torch_zp)
        assert (ours.qmin, ours.qmax) == (0, 127)  # reduce_range == 7-bit unsigned

    def test_searched_range_within_raw_extent(self) -> None:
        bounds = extract_histogram_bounds(self._calibrated_observer())
        assert bounds.raw_low is not None and bounds.raw_high is not None
        assert bounds.raw_low - 1e-6 <= bounds.low
        assert bounds.high <= bounds.raw_high + 1e-6

    def test_rejects_wrong_observer_type(self) -> None:
        with pytest.raises(TypeError, match="HistogramObserver"):
            extract_histogram_bounds(PerChannelMinMaxObserver())  # type: ignore[arg-type]


class TestFrozenFakeQuant:
    def test_matches_core_and_preserves_dtype(self) -> None:
        params = compute_quant_params(np.array([0.0, 4.0], dtype=np.float32), bits=7, signed=False)
        module = FrozenQuantScopeFakeQuant(params)
        x = torch.rand(2, 3) * 4.0
        out = module(x)
        np.testing.assert_array_equal(out.numpy(), fake_quantize(x.numpy(), params))
        assert out.dtype == x.dtype and out.shape == x.shape

    def test_never_updates_state(self) -> None:
        params = compute_quant_params(np.array([0.0, 1.0], dtype=np.float32))
        module = FrozenQuantScopeFakeQuant(params)
        before = module.params
        module(torch.rand(8) * 100.0)  # far outside the frozen range
        assert module.params is before  # no recalibration of any kind


class TestPrimitiveParity:
    """QuantScope round/clamp/dequant vs Torch's quantize ops, identical qparams."""

    CASES: typing.ClassVar[dict[str, np.ndarray]] = {
        "halfway": np.array([0.05, 0.15, 0.25, -0.05, -0.15], dtype=np.float32),
        "saturation": np.array([-100.0, 100.0, 12.7, -12.8], dtype=np.float32),
        "zero": np.zeros(4, dtype=np.float32),
        "constant": np.full(4, 0.37, dtype=np.float32),
        "negative": np.linspace(-1.0, 0.0, 9).astype(np.float32),
    }

    @pytest.mark.parametrize("name", sorted(CASES))
    def test_per_tensor_affine_codes_and_dequant(self, name: str) -> None:
        data = self.CASES[name]
        scale, zp = 0.1, 3
        params = compute_quant_params(
            np.array([-12.8, 12.7], dtype=np.float32), bits=8, signed=True
        )
        # Force the exact torch qparams onto our metadata for the primitive test.
        from quantscope.quantization.affine import QuantParams

        params = QuantParams(
            scale=np.asarray(scale, dtype=np.float32),
            zero_point=np.asarray(zp, dtype=np.int32),
            qmin=-128,
            qmax=127,
            bits=8,
            signed=True,
            scheme=Scheme.ASYMMETRIC,
            granularity=Granularity.PER_TENSOR,
        )
        torch_q = torch.quantize_per_tensor(torch.from_numpy(data), scale, zp, torch.qint8)
        ours = quantize(data, params)
        theirs = torch_q.int_repr().numpy().astype(np.int32)

        # Documented arithmetic-precision boundary (ADR-011): Torch divides
        # in float32, QuantScope in float64. They may disagree by exactly
        # one code, and only where the float32 quotient lands on an exact
        # .5 rounding tie.
        quotient_f32 = (data.astype(np.float32) / np.float32(scale)).astype(np.float32)
        tie = np.abs(quotient_f32 - np.floor(quotient_f32) - 0.5) == 0.0
        diff = ours != theirs
        assert not np.any(diff & ~tie), "disagreement outside float32 tie boundary"
        assert np.all(np.abs(ours - theirs)[diff] == 1)
        np.testing.assert_allclose(
            dequantize(ours, params)[~diff],
            torch_q.dequantize().numpy()[~diff],
            rtol=1e-6,
            atol=1e-7,
        )

    def test_per_channel_symmetric_codes(self) -> None:
        torch.manual_seed(2)
        data = torch.randn(4, 5).numpy().astype(np.float32)
        scales = np.array([0.02, 0.05, 0.1, 0.4], dtype=np.float32)
        zps = np.zeros(4, dtype=np.int32)
        from quantscope.quantization.affine import QuantParams

        params = QuantParams(
            scale=scales,
            zero_point=zps,
            qmin=-128,
            qmax=127,
            bits=8,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        torch_q = torch.quantize_per_channel(
            torch.from_numpy(data),
            torch.from_numpy(scales.astype(np.float64)),
            torch.from_numpy(zps.astype(np.int64)),
            0,
            torch.qint8,
        )
        np.testing.assert_array_equal(quantize(data, params), torch_q.int_repr().numpy())


class TestWeightBoundsExtraction:
    def test_fused_conv_weight_bounds(self) -> None:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx

        from quantscope.models.bottleneck_resnet import BottleneckResNet

        torch.backends.quantized.engine = "fbgemm"
        model = BottleneckResNet(num_classes=4, bottleneck_width=4).eval()
        prepared = prepare_fx(
            model, get_default_qconfig_mapping("fbgemm"), (torch.randn(1, 1, 32, 32),)
        )
        stem = prepared.get_submodule("stem_conv")
        bounds = extract_weight_bounds(stem)
        assert bounds.granularity is Granularity.PER_CHANNEL
        assert bounds.channel_axis == 0
        assert (bounds.quant_min, bounds.quant_max) == (-128, 127)
        assert bounds.low.shape == (16,)  # one range per output channel
