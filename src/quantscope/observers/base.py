"""Observer base class: streaming calibration statistics -> QuantParams.

Observers watch batches of real values during calibration and decide the
quantization range. They are backend-independent (numpy-only, ADR-002);
torch adapters wrap them where needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from quantscope.quantization import (
    Granularity,
    QuantParams,
    Scheme,
    compute_quant_params,
)

__all__ = ["CalibrationObserver", "MinMaxObserver"]


class CalibrationObserver(ABC):
    """Streaming observer accumulating statistics over calibration batches.

    Subclasses implement :meth:`update` to accumulate statistics and
    :meth:`_calibration_values` to expose the representative value range
    used for parameter computation.
    """

    def __init__(
        self,
        *,
        bits: int = 8,
        signed: bool = True,
        scheme: Scheme = Scheme.ASYMMETRIC,
        granularity: Granularity = Granularity.PER_TENSOR,
        channel_axis: int | None = None,
        narrow_range: bool = False,
    ) -> None:
        self.bits = bits
        self.signed = signed
        self.scheme = scheme
        self.granularity = granularity
        self.channel_axis = channel_axis
        self.narrow_range = narrow_range
        self._num_batches = 0

    @property
    def num_batches(self) -> int:
        """Number of calibration batches observed so far."""
        return self._num_batches

    def observe(self, values: np.ndarray) -> None:
        """Validate and accumulate one calibration batch."""
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            raise ValueError("cannot observe an empty batch")
        if not np.all(np.isfinite(values)):
            raise ValueError("calibration batch contains NaN or infinite values")
        self.update(values)
        self._num_batches += 1

    @abstractmethod
    def update(self, values: np.ndarray) -> None:
        """Accumulate statistics from one validated batch."""

    @abstractmethod
    def _calibration_values(self) -> np.ndarray:
        """Return values representing the calibrated range (e.g. [low, high])."""

    def to_quant_params(self) -> QuantParams:
        """Produce quantization parameters from accumulated statistics.

        Raises:
            RuntimeError: if no calibration data was observed.
        """
        if self._num_batches == 0:
            raise RuntimeError(
                f"{type(self).__name__} has observed no calibration data; "
                "call observe() with at least one batch first"
            )
        values = self._calibration_values()
        if self.granularity is Granularity.PER_CHANNEL:
            # _calibration_values returns a stacked (2, C) low/high array, so
            # the channel axis *here* is the last one; the resulting params
            # are rebuilt to carry the data-layout channel axis instead.
            params = compute_quant_params(
                values,
                bits=self.bits,
                signed=self.signed,
                scheme=self.scheme,
                granularity=self.granularity,
                channel_axis=-1,
                narrow_range=self.narrow_range,
            )
            return QuantParams(
                scale=params.scale,
                zero_point=params.zero_point,
                qmin=params.qmin,
                qmax=params.qmax,
                bits=params.bits,
                signed=params.signed,
                scheme=params.scheme,
                granularity=params.granularity,
                channel_axis=self.channel_axis,
            )
        return compute_quant_params(
            values,
            bits=self.bits,
            signed=self.signed,
            scheme=self.scheme,
            granularity=self.granularity,
            channel_axis=None,
            narrow_range=self.narrow_range,
        )


class MinMaxObserver(CalibrationObserver):
    """Baseline observer tracking the global running min/max.

    Included as the comparison baseline for the custom observers; matches
    the behavior of standard min-max calibration.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._low: np.ndarray | None = None
        self._high: np.ndarray | None = None

    def _reduce(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.granularity is Granularity.PER_CHANNEL:
            if self.channel_axis is None:
                raise ValueError("per-channel observer requires channel_axis")
            axis = self.channel_axis % values.ndim
            reduce_axes = tuple(ax for ax in range(values.ndim) if ax != axis)
            return values.min(axis=reduce_axes), values.max(axis=reduce_axes)
        return np.asarray(values.min()), np.asarray(values.max())

    def update(self, values: np.ndarray) -> None:
        low, high = self._reduce(values)
        if self._low is None or self._high is None:
            self._low, self._high = low, high
        else:
            self._low = np.minimum(self._low, low)
            self._high = np.maximum(self._high, high)

    def _calibration_values(self) -> np.ndarray:
        assert self._low is not None and self._high is not None
        return np.stack([self._low, self._high])
