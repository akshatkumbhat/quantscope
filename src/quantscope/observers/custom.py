"""Custom calibration observers.

Three strategies beyond plain min-max:

- :class:`PercentileClippingObserver` — outlier-resistant percentile ranges.
- :class:`PowerOfTwoScaleObserver` — shift-friendly power-of-two scales.
- :class:`MSEGridSearchObserver` — clipping threshold minimizing MSE.

Percentile and MSE observers need a sample of the observed distribution,
not just min/max. They keep a bounded, deterministic subsample per batch
(``samples_per_batch``); memory grows linearly with the number of
calibration batches, which is small by design (see PROJECT_SPEC testing
rules). Both support per-tensor granularity only — per-channel percentile/
MSE calibration is out of scope and rejected explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantscope.analysis import mse
from quantscope.observers.base import CalibrationObserver, MinMaxObserver
from quantscope.quantization import (
    Granularity,
    QuantParams,
    compute_quant_params,
    fake_quantize,
    power_of_two_scale,
)

__all__ = [
    "MSEGridSearchObserver",
    "PercentileClippingObserver",
    "PowerOfTwoScaleObserver",
]

_DEFAULT_SAMPLES_PER_BATCH = 8192
_SAMPLING_SEED = 0x5EED  # fixed: calibration must be reproducible


class _SamplingObserver(CalibrationObserver):
    """Shared machinery: keep a deterministic subsample of each batch."""

    def __init__(
        self, *, samples_per_batch: int = _DEFAULT_SAMPLES_PER_BATCH, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if self.granularity is not Granularity.PER_TENSOR:
            raise ValueError(f"{type(self).__name__} supports per-tensor granularity only")
        if samples_per_batch < 1:
            raise ValueError("samples_per_batch must be >= 1")
        self._samples_per_batch = samples_per_batch
        self._rng = np.random.default_rng(_SAMPLING_SEED)
        self._samples: list[np.ndarray] = []

    def update(self, values: np.ndarray) -> None:
        flat = values.ravel()
        if flat.size > self._samples_per_batch:
            idx = self._rng.choice(flat.size, size=self._samples_per_batch, replace=False)
            flat = flat[idx]
        self._samples.append(flat.copy())

    def _all_samples(self) -> np.ndarray:
        return np.concatenate(self._samples)


class PercentileClippingObserver(_SamplingObserver):
    """Clips the calibration range to percentiles of the observed values.

    Min-max calibration lets a single outlier stretch the range and waste
    integer codes. Clipping to e.g. the [0.1, 99.9] percentiles trades a
    small amount of saturation on outliers for much finer resolution on
    the bulk of the distribution.
    """

    def __init__(
        self,
        *,
        lower_percentile: float = 0.1,
        upper_percentile: float = 99.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
            raise ValueError(
                "percentiles must satisfy 0 <= lower < upper <= 100, got "
                f"({lower_percentile}, {upper_percentile})"
            )
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def _calibration_values(self) -> np.ndarray:
        samples = self._all_samples()
        low, high = np.percentile(samples, [self.lower_percentile, self.upper_percentile])
        return np.array([low, high], dtype=np.float32)


class PowerOfTwoScaleObserver(MinMaxObserver):
    """Min-max calibration with the scale snapped to a power of two.

    Power-of-two scales let hardware rescale with bit shifts instead of
    multipliers. The scale is rounded **up** to the next power of two by
    default so the calibrated range stays fully representable (no snapping-
    induced clipping); steps become at most 2x coarser.
    """

    def __init__(self, *, rounding: str = "up", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rounding = rounding
        # Validate the mode eagerly rather than failing after calibration.
        power_of_two_scale(1.0, mode=rounding)

    def to_quant_params(self) -> QuantParams:
        params = super().to_quant_params()
        pot_scale = power_of_two_scale(params.scale, mode=self.rounding)
        if params.scheme.value == "asymmetric":
            # Zero point must be recomputed against the snapped scale so
            # zero stays exactly representable.
            low = params.scale.astype(np.float64) * (params.qmin - params.zero_point)
            zero_point = np.clip(
                np.round(params.qmin - low / pot_scale), params.qmin, params.qmax
            ).astype(np.int32)
        else:
            zero_point = params.zero_point
        return QuantParams(
            scale=pot_scale,
            zero_point=zero_point,
            qmin=params.qmin,
            qmax=params.qmax,
            bits=params.bits,
            signed=params.signed,
            scheme=params.scheme,
            granularity=params.granularity,
            channel_axis=params.channel_axis,
        )


class MSEGridSearchObserver(_SamplingObserver):
    """Grid-searches the clipping range that minimizes quantization MSE.

    Candidate ranges shrink the observed min/max by factors in
    ``[min_clip_fraction, 1.0]``; for each candidate the sampled values are
    fake-quantized and the reconstruction MSE measured. The best candidate
    wins. More expensive than percentile clipping but directly optimizes
    the metric that matters.
    """

    def __init__(
        self, *, num_candidates: int = 32, min_clip_fraction: float = 0.3, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if not 0.0 < min_clip_fraction <= 1.0:
            raise ValueError("min_clip_fraction must be in (0, 1]")
        self.num_candidates = num_candidates
        self.min_clip_fraction = min_clip_fraction

    def _calibration_values(self) -> np.ndarray:
        samples = self._all_samples()
        low = float(samples.min())
        high = float(samples.max())
        if high - low <= np.finfo(np.float32).eps and abs(high) <= np.finfo(np.float32).eps:
            # Degenerate all-zero/constant-at-zero data: nothing to search.
            return np.array([low, high], dtype=np.float32)

        best_range = np.array([low, high], dtype=np.float32)
        best_mse = np.inf
        for fraction in np.linspace(self.min_clip_fraction, 1.0, self.num_candidates):
            candidate = np.array([low * fraction, high * fraction], dtype=np.float32)
            if candidate[1] - candidate[0] <= np.finfo(np.float32).eps:
                continue
            params = compute_quant_params(
                candidate,
                bits=self.bits,
                signed=self.signed,
                scheme=self.scheme,
                granularity=Granularity.PER_TENSOR,
                narrow_range=self.narrow_range,
            )
            candidate_mse = mse(samples, fake_quantize(samples, params))
            if candidate_mse < best_mse:
                best_mse = candidate_mse
                best_range = candidate
        return best_range
