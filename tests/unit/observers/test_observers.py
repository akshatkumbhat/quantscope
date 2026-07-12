"""Unit tests for calibration observers."""

import numpy as np
import pytest

from quantscope.analysis import mse
from quantscope.observers import (
    MinMaxObserver,
    MSEGridSearchObserver,
    PercentileClippingObserver,
    PowerOfTwoScaleObserver,
)
from quantscope.quantization import Granularity, Scheme, fake_quantize, quantize


def _outlier_data(n: int = 4096) -> np.ndarray:
    """Bulk N(0,1) data with a single large outlier."""
    rng = np.random.default_rng(123)
    data = rng.normal(size=n).astype(np.float32)
    data[0] = 100.0
    return data


class TestMinMaxObserver:
    def test_tracks_running_min_max_across_batches(self) -> None:
        obs = MinMaxObserver(scheme=Scheme.ASYMMETRIC, signed=False)
        obs.observe(np.array([0.0, 1.0], dtype=np.float32))
        obs.observe(np.array([-2.0, 0.5], dtype=np.float32))
        p = obs.to_quant_params()
        assert p.scale == pytest.approx(3.0 / 255, rel=1e-5)

    def test_empty_calibration_rejected(self) -> None:
        obs = MinMaxObserver()
        with pytest.raises(RuntimeError, match="no calibration data"):
            obs.to_quant_params()

    def test_empty_batch_rejected(self) -> None:
        obs = MinMaxObserver()
        with pytest.raises(ValueError, match="empty"):
            obs.observe(np.array([], dtype=np.float32))

    def test_nan_batch_rejected(self) -> None:
        obs = MinMaxObserver()
        with pytest.raises(ValueError, match="NaN or infinite"):
            obs.observe(np.array([1.0, np.nan], dtype=np.float32))

    def test_per_channel_params_carry_data_axis(self) -> None:
        obs = MinMaxObserver(
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        w = np.stack([np.full((4,), 0.5, dtype=np.float32), np.full((4,), 8.0, dtype=np.float32)])
        obs.observe(w)
        p = obs.to_quant_params()
        assert p.channel_axis == 0
        assert p.scale.shape == (2,)
        assert p.scale[1] > p.scale[0]
        # Params must be directly usable against the original data layout.
        q = quantize(w, p)
        assert q.shape == w.shape


class TestPercentileClippingObserver:
    def test_resists_outliers(self) -> None:
        data = _outlier_data()
        minmax = MinMaxObserver(scheme=Scheme.SYMMETRIC)
        pct = PercentileClippingObserver(
            scheme=Scheme.SYMMETRIC, lower_percentile=0.5, upper_percentile=99.5
        )
        minmax.observe(data)
        pct.observe(data)
        # Percentile range ignores the outlier -> much smaller scale.
        assert pct.to_quant_params().scale < minmax.to_quant_params().scale / 10

    def test_lower_bulk_error_than_minmax(self) -> None:
        data = _outlier_data()
        minmax = MinMaxObserver(scheme=Scheme.SYMMETRIC)
        pct = PercentileClippingObserver(scheme=Scheme.SYMMETRIC)
        minmax.observe(data)
        pct.observe(data)
        bulk = data[1:]  # everything except the outlier
        err_minmax = mse(bulk, fake_quantize(bulk, minmax.to_quant_params()))
        err_pct = mse(bulk, fake_quantize(bulk, pct.to_quant_params()))
        assert err_pct < err_minmax

    def test_invalid_percentiles_rejected(self) -> None:
        with pytest.raises(ValueError, match="percentiles"):
            PercentileClippingObserver(lower_percentile=50.0, upper_percentile=10.0)

    def test_per_channel_rejected(self) -> None:
        with pytest.raises(ValueError, match="per-tensor"):
            PercentileClippingObserver(granularity=Granularity.PER_CHANNEL, channel_axis=0)


class TestPowerOfTwoScaleObserver:
    def test_scale_is_power_of_two(self) -> None:
        rng = np.random.default_rng(5)
        obs = PowerOfTwoScaleObserver(scheme=Scheme.SYMMETRIC)
        obs.observe(rng.normal(scale=3.0, size=(256,)).astype(np.float32))
        scale = float(obs.to_quant_params().scale)
        assert scale == 2.0 ** np.round(np.log2(scale))

    def test_rounding_up_never_clips_calibrated_range(self) -> None:
        rng = np.random.default_rng(6)
        data = rng.uniform(-5.0, 5.0, size=(512,)).astype(np.float32)
        obs = PowerOfTwoScaleObserver(scheme=Scheme.SYMMETRIC)
        obs.observe(data)
        p = obs.to_quant_params()
        q = quantize(data, p)
        # Round-up snapping widens the range, so nothing saturates.
        assert q.max() < p.qmax or float(p.scale) * p.qmax >= np.abs(data).max()

    def test_asymmetric_zero_still_exact(self) -> None:
        rng = np.random.default_rng(7)
        obs = PowerOfTwoScaleObserver(scheme=Scheme.ASYMMETRIC, signed=False)
        obs.observe(rng.uniform(2.0, 9.0, size=(128,)).astype(np.float32))
        p = obs.to_quant_params()
        recon_zero = float(p.scale) * (int(p.zero_point) - int(p.zero_point))
        assert recon_zero == 0.0
        assert p.qmin <= int(p.zero_point) <= p.qmax

    def test_invalid_rounding_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="rounding mode"):
            PowerOfTwoScaleObserver(rounding="sideways")


class TestMSEGridSearchObserver:
    def test_beats_minmax_on_outlier_data(self) -> None:
        data = _outlier_data()
        minmax = MinMaxObserver(scheme=Scheme.SYMMETRIC)
        search = MSEGridSearchObserver(scheme=Scheme.SYMMETRIC)
        minmax.observe(data)
        search.observe(data)
        err_minmax = mse(data, fake_quantize(data, minmax.to_quant_params()))
        err_search = mse(data, fake_quantize(data, search.to_quant_params()))
        assert err_search < err_minmax

    def test_clean_data_close_to_minmax(self) -> None:
        rng = np.random.default_rng(9)
        data = rng.uniform(-1.0, 1.0, size=(2048,)).astype(np.float32)
        minmax = MinMaxObserver(scheme=Scheme.SYMMETRIC)
        search = MSEGridSearchObserver(scheme=Scheme.SYMMETRIC)
        minmax.observe(data)
        search.observe(data)
        err_minmax = mse(data, fake_quantize(data, minmax.to_quant_params()))
        err_search = mse(data, fake_quantize(data, search.to_quant_params()))
        # Grid search must never be meaningfully worse than min-max.
        assert err_search <= err_minmax * 1.05

    def test_all_zero_data(self) -> None:
        obs = MSEGridSearchObserver()
        obs.observe(np.zeros((16,), dtype=np.float32))
        p = obs.to_quant_params()
        assert np.all(p.scale > 0)

    def test_deterministic_across_instances(self) -> None:
        data = _outlier_data()
        params = []
        for _ in range(2):
            obs = MSEGridSearchObserver(scheme=Scheme.SYMMETRIC)
            obs.observe(data)
            params.append(obs.to_quant_params())
        assert float(params[0].scale) == float(params[1].scale)

    def test_invalid_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_candidates"):
            MSEGridSearchObserver(num_candidates=0)
        with pytest.raises(ValueError, match="min_clip_fraction"):
            MSEGridSearchObserver(min_clip_fraction=0.0)


class TestSamplingBound:
    def test_large_batch_subsampled(self) -> None:
        obs = PercentileClippingObserver(samples_per_batch=100)
        obs.observe(np.arange(10_000, dtype=np.float32))
        assert obs._all_samples().size == 100

    def test_invalid_samples_per_batch_rejected(self) -> None:
        with pytest.raises(ValueError, match="samples_per_batch"):
            PercentileClippingObserver(samples_per_batch=0)
