"""Unit tests for the backend-independent affine quantization core."""

import numpy as np
import pytest

from quantscope.quantization import (
    Granularity,
    QuantParams,
    Scheme,
    compute_quant_params,
    dequantize,
    fake_quantize,
    integer_range,
    power_of_two_scale,
    quantize,
)


class TestIntegerRange:
    def test_signed_int8(self) -> None:
        r = integer_range(8, signed=True)
        assert (r.qmin, r.qmax) == (-128, 127)

    def test_signed_int8_narrow(self) -> None:
        r = integer_range(8, signed=True, narrow_range=True)
        assert (r.qmin, r.qmax) == (-127, 127)

    def test_unsigned_uint8(self) -> None:
        r = integer_range(8, signed=False)
        assert (r.qmin, r.qmax) == (0, 255)

    def test_int4(self) -> None:
        r = integer_range(4, signed=True)
        assert (r.qmin, r.qmax) == (-8, 7)

    def test_uint4(self) -> None:
        r = integer_range(4, signed=False)
        assert (r.qmin, r.qmax) == (0, 15)

    @pytest.mark.parametrize("bits", [0, 1, 17, 64, -8])
    def test_invalid_bit_widths(self, bits: int) -> None:
        with pytest.raises(ValueError, match="bits"):
            integer_range(bits, signed=True)

    def test_non_integer_bits_rejected(self) -> None:
        with pytest.raises(ValueError, match="bits"):
            integer_range(8.0, signed=True)  # type: ignore[arg-type]

    def test_unsigned_narrow_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="narrow_range"):
            integer_range(8, signed=False, narrow_range=True)


class TestComputeQuantParams:
    def test_symmetric_per_tensor(self) -> None:
        values = np.array([-2.0, 0.5, 1.0], dtype=np.float32)
        p = compute_quant_params(values, bits=8, signed=True, scheme=Scheme.SYMMETRIC)
        assert p.zero_point == 0
        # Symmetric convention: scale = max|x| / qmax (PyTorch-compatible).
        assert p.scale == pytest.approx(2.0 / 127, rel=1e-6)

    def test_asymmetric_per_tensor(self) -> None:
        values = np.array([0.0, 10.0], dtype=np.float32)
        p = compute_quant_params(values, bits=8, signed=False, scheme=Scheme.ASYMMETRIC)
        assert p.scale == pytest.approx(10.0 / 255, rel=1e-6)
        assert p.zero_point == 0  # range already anchored at zero

    def test_asymmetric_range_includes_zero(self) -> None:
        # All-positive data: range must be widened to include zero.
        values = np.array([5.0, 10.0], dtype=np.float32)
        p = compute_quant_params(values, bits=8, signed=False, scheme=Scheme.ASYMMETRIC)
        assert dequantize(np.asarray(p.zero_point), p) == pytest.approx(0.0)

    def test_per_channel_shapes(self) -> None:
        rng = np.random.default_rng(0)
        w = rng.normal(size=(4, 3, 2, 2)).astype(np.float32)
        p = compute_quant_params(
            w,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        assert p.scale.shape == (4,)
        assert p.zero_point.shape == (4,)
        # Each channel's scale reflects that channel's max magnitude.
        expected = np.max(np.abs(w.reshape(4, -1)), axis=1) / 127
        np.testing.assert_allclose(p.scale, expected, rtol=1e-6)

    def test_per_channel_negative_axis(self) -> None:
        w = np.ones((2, 3), dtype=np.float32)
        p = compute_quant_params(w, granularity=Granularity.PER_CHANNEL, channel_axis=-1)
        assert p.scale.shape == (3,)
        assert p.channel_axis == 1

    def test_constant_tensor(self) -> None:
        values = np.full((3, 3), 7.5, dtype=np.float32)
        p = compute_quant_params(values, scheme=Scheme.SYMMETRIC)
        assert np.all(p.scale > 0)
        # Constant magnitude is representable within a rounding step.
        recon = fake_quantize(values, p)
        assert np.max(np.abs(recon - values)) <= p.scale / 2 + 1e-6

    def test_all_zero_tensor(self) -> None:
        values = np.zeros((4,), dtype=np.float32)
        for scheme in Scheme:
            p = compute_quant_params(values, scheme=scheme)
            assert np.all(p.scale > 0)
            np.testing.assert_array_equal(fake_quantize(values, p), values)

    def test_narrow_range_values(self) -> None:
        values = np.array([1.0, 1.0 + 1e-9], dtype=np.float32)
        p = compute_quant_params(values, scheme=Scheme.ASYMMETRIC)
        assert np.all(p.scale > 0)
        assert np.all(np.isfinite(p.scale))

    def test_empty_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_quant_params(np.array([], dtype=np.float32))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nan_inf_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            compute_quant_params(np.array([1.0, bad], dtype=np.float32))

    def test_missing_channel_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="channel_axis"):
            compute_quant_params(
                np.ones((2, 2), dtype=np.float32),
                granularity=Granularity.PER_CHANNEL,
            )

    def test_out_of_range_channel_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="channel_axis"):
            compute_quant_params(
                np.ones((2, 2), dtype=np.float32),
                granularity=Granularity.PER_CHANNEL,
                channel_axis=5,
            )

    def test_channel_axis_with_per_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="channel_axis"):
            compute_quant_params(np.ones((2, 2), dtype=np.float32), channel_axis=0)


class TestQuantizeDequantize:
    def test_saturation_at_bounds(self) -> None:
        values = np.array([-1.0, 1.0], dtype=np.float32)
        p = compute_quant_params(values, bits=8, signed=True, scheme=Scheme.SYMMETRIC)
        q = quantize(np.array([-100.0, 100.0], dtype=np.float32), p)
        np.testing.assert_array_equal(q, [p.qmin, p.qmax])

    def test_round_trip_error_bound(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(-3.0, 5.0, size=(64,)).astype(np.float32)
        p = compute_quant_params(values, bits=8, scheme=Scheme.ASYMMETRIC, signed=True)
        recon = fake_quantize(values, p)
        # In-range values reconstruct within half a quantization step.
        assert np.max(np.abs(recon - values)) <= float(p.scale) / 2 + 1e-6

    def test_int4_coarser_than_int8(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(size=(256,)).astype(np.float32)
        err = {}
        for bits in (4, 8):
            p = compute_quant_params(values, bits=bits, scheme=Scheme.SYMMETRIC)
            err[bits] = float(np.mean((fake_quantize(values, p) - values) ** 2))
        assert err[4] > err[8]

    def test_per_channel_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        # Channels with wildly different ranges — per-channel must adapt.
        w = np.stack([rng.normal(scale=s, size=(8, 3, 3)) for s in (0.01, 1.0, 100.0)])
        w = w.astype(np.float32)
        p = compute_quant_params(
            w,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        recon = fake_quantize(w, p)
        for c in range(3):
            step = float(p.scale[c])
            assert np.max(np.abs(recon[c] - w[c])) <= step / 2 + 1e-6

    def test_quantize_output_dtype_and_range(self) -> None:
        values = np.linspace(-1, 1, 11).astype(np.float32)
        p = compute_quant_params(values, bits=4, signed=True)
        q = quantize(values, p)
        assert q.dtype == np.int32
        assert q.min() >= p.qmin and q.max() <= p.qmax

    def test_quantize_channel_mismatch_rejected(self) -> None:
        p = compute_quant_params(
            np.ones((3, 2), dtype=np.float32),
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        with pytest.raises(ValueError, match="channels"):
            quantize(np.ones((4, 2), dtype=np.float32), p)

    def test_quantize_nan_rejected(self) -> None:
        p = compute_quant_params(np.array([1.0], dtype=np.float32))
        with pytest.raises(ValueError, match="NaN or infinite"):
            quantize(np.array([np.nan], dtype=np.float32), p)

    def test_dequantize_out_of_range_rejected(self) -> None:
        p = compute_quant_params(np.array([1.0], dtype=np.float32), bits=4, signed=True)
        with pytest.raises(ValueError, match="outside"):
            dequantize(np.array([100]), p)


class TestPowerOfTwoScale:
    def test_exact_powers_unchanged(self) -> None:
        scales = np.array([0.25, 0.5, 1.0, 2.0, 8.0])
        np.testing.assert_allclose(power_of_two_scale(scales), scales)

    def test_rounds_to_nearest_power(self) -> None:
        assert power_of_two_scale(0.3) == pytest.approx(0.25)
        assert power_of_two_scale(0.4) == pytest.approx(0.5)
        assert power_of_two_scale(3.0) == pytest.approx(4.0)  # log2(3)≈1.58 → 2

    def test_within_sqrt2_factor(self) -> None:
        rng = np.random.default_rng(11)
        scales = 10.0 ** rng.uniform(-6, 3, size=100)
        approx = power_of_two_scale(scales)
        ratio = approx / scales
        assert np.all(ratio <= np.sqrt(2) + 1e-9)
        assert np.all(ratio >= 1 / np.sqrt(2) - 1e-9)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_scales_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="scale"):
            power_of_two_scale(bad)


class TestQuantParamsValidation:
    def test_negative_scale_rejected(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            QuantParams(
                scale=np.asarray(-1.0),
                zero_point=np.asarray(0),
                qmin=-128,
                qmax=127,
                bits=8,
                signed=True,
                scheme=Scheme.SYMMETRIC,
                granularity=Granularity.PER_TENSOR,
            )

    def test_zero_point_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero_point"):
            QuantParams(
                scale=np.asarray(1.0),
                zero_point=np.asarray(300),
                qmin=-128,
                qmax=127,
                bits=8,
                signed=True,
                scheme=Scheme.ASYMMETRIC,
                granularity=Granularity.PER_TENSOR,
            )

    def test_per_channel_requires_axis(self) -> None:
        with pytest.raises(ValueError, match="channel_axis"):
            QuantParams(
                scale=np.ones(3),
                zero_point=np.zeros(3, dtype=np.int32),
                qmin=-128,
                qmax=127,
                bits=8,
                signed=True,
                scheme=Scheme.SYMMETRIC,
                granularity=Granularity.PER_CHANNEL,
            )
