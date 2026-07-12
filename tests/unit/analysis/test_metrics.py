"""Unit tests for numerical-error metrics."""

import numpy as np
import pytest

from quantscope.analysis import (
    compare,
    cosine_similarity,
    max_abs_error,
    mse,
    saturation_rate,
    sqnr_db,
)


class TestMse:
    def test_identical_tensors(self) -> None:
        x = np.array([1.0, -2.0, 3.0])
        assert mse(x, x) == 0.0

    def test_known_value(self) -> None:
        assert mse(np.array([0.0, 0.0]), np.array([1.0, -1.0])) == pytest.approx(1.0)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            mse(np.zeros(3), np.zeros(4))

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mse(np.array([]), np.array([]))

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            mse(np.array([np.nan]), np.array([1.0]))


class TestSqnr:
    def test_perfect_reconstruction_is_inf(self) -> None:
        x = np.array([1.0, 2.0])
        assert sqnr_db(x, x) == float("inf")

    def test_zero_signal_nonzero_noise_is_neg_inf(self) -> None:
        assert sqnr_db(np.zeros(4), np.ones(4)) == float("-inf")

    def test_zero_signal_zero_noise_is_inf(self) -> None:
        assert sqnr_db(np.zeros(4), np.zeros(4)) == float("inf")

    def test_known_value(self) -> None:
        # signal power 1, noise power 0.01 -> 20 dB
        ref = np.array([1.0, -1.0])
        approx = np.array([1.1, -0.9])
        assert sqnr_db(ref, approx) == pytest.approx(20.0, abs=1e-9)

    def test_more_bits_higher_sqnr(self) -> None:
        from quantscope.quantization import Scheme, compute_quant_params, fake_quantize

        rng = np.random.default_rng(0)
        x = rng.normal(size=(512,)).astype(np.float32)
        sqnrs = {}
        for bits in (4, 8):
            p = compute_quant_params(x, bits=bits, scheme=Scheme.SYMMETRIC)
            sqnrs[bits] = sqnr_db(x, fake_quantize(x, p))
        assert sqnrs[8] > sqnrs[4] + 15  # ~6 dB/bit theoretical spacing


class TestCosine:
    def test_identical_is_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(x, x) == pytest.approx(1.0)

    def test_opposite_is_minus_one(self) -> None:
        x = np.array([1.0, 2.0])
        assert cosine_similarity(x, -x) == pytest.approx(-1.0)

    def test_orthogonal_is_zero(self) -> None:
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_both_zero_defined_as_one(self) -> None:
        assert cosine_similarity(np.zeros(3), np.zeros(3)) == 1.0

    def test_one_zero_defined_as_zero(self) -> None:
        assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0


class TestMaxAbsError:
    def test_known_value(self) -> None:
        ref = np.array([0.0, 5.0, -3.0])
        approx = np.array([0.5, 5.0, -1.0])
        assert max_abs_error(ref, approx) == pytest.approx(2.0)


class TestSaturationRate:
    def test_no_saturation(self) -> None:
        q = np.array([-100, 0, 100])
        assert saturation_rate(q, -128, 127) == 0.0

    def test_full_saturation(self) -> None:
        q = np.array([-128, 127, 127, -128])
        assert saturation_rate(q, -128, 127) == 1.0

    def test_partial(self) -> None:
        q = np.array([-128, 0, 0, 127])
        assert saturation_rate(q, -128, 127) == pytest.approx(0.5)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            saturation_rate(np.array([]), -128, 127)

    def test_invalid_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="qmin"):
            saturation_rate(np.array([0]), 127, -128)


class TestCompare:
    def test_full_metric_set(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.normal(size=(32,))
        approx = ref + rng.normal(scale=0.01, size=(32,))
        m = compare(ref, approx)
        assert m.mse > 0
        assert m.sqnr_db > 20
        assert m.cosine > 0.99
        assert m.max_abs_error > 0
        assert m.num_elements == 32

    def test_to_dict_serializable(self) -> None:
        import json

        x = np.array([1.0, 2.0])
        d = compare(x, x * 1.01).to_dict()
        assert set(d) == {"mse", "sqnr_db", "cosine", "max_abs_error", "num_elements"}
        json.dumps(d)  # must not raise
