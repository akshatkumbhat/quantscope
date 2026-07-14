"""Unit tests for the ADR-012 impulse stress mechanism and observer plumbing."""

import numpy as np
import pytest
import torch

from quantscope.data.texture10 import (
    Texture10Params,
    apply_impulse_stress,
    make_texture10,
)
from quantscope.models.bottleneck_resnet import BottleneckResNet
from quantscope.observers import MinMaxObserver, PercentileClippingObserver
from quantscope.quantization.simulate import (
    SimQuantConfig,
    calibrate_activation_params,
    simulate_quantized,
)

_SMALL = Texture10Params(num_classes=4, image_size=16)


def _clean(n: int = 32, seed: int = 0):
    return make_texture10(num_samples=n, seed=seed, params=_SMALL)


class TestImpulseStress:
    def test_pairing_preserved(self) -> None:
        clean = _clean()
        stressed = apply_impulse_stress(clean, seed=5)
        # Labels identical; images identical everywhere except impulses.
        assert torch.equal(clean.tensors[1], stressed.tensors[1])
        diff = (clean.tensors[0] != stressed.tensors[0]).float()
        per_image = diff.reshape(diff.shape[0], -1).sum(dim=1)
        expected = max(1, round(0.002 * 16 * 16))
        assert torch.all(per_image <= expected)  # only impulse pixels changed

    def test_deterministic(self) -> None:
        clean = _clean()
        a = apply_impulse_stress(clean, seed=7)
        b = apply_impulse_stress(clean, seed=7)
        assert torch.equal(a.tensors[0], b.tensors[0])

    def test_magnitude_and_balanced_signs(self) -> None:
        clean = _clean(n=16)
        stressed = apply_impulse_stress(clean, fraction=0.02, magnitude=6.0, seed=1)
        changed = clean.tensors[0] != stressed.tensors[0]
        for i in range(16):
            std = float(clean.tensors[0][i].std())
            vals = stressed.tensors[0][i][changed[i]]
            np.testing.assert_allclose(np.abs(vals.numpy()), 6.0 * std, rtol=0.15)
            n_pos = int((vals > 0).sum())
            assert abs(n_pos - (vals.numel() - n_pos)) <= 1  # balanced signs

    def test_original_untouched(self) -> None:
        clean = _clean()
        before = clean.tensors[0].clone()
        apply_impulse_stress(clean, seed=2)
        assert torch.equal(clean.tensors[0], before)

    def test_invalid_params_rejected(self) -> None:
        clean = _clean()
        with pytest.raises(ValueError, match="fraction"):
            apply_impulse_stress(clean, fraction=0.0)
        with pytest.raises(ValueError, match="magnitude"):
            apply_impulse_stress(clean, magnitude=-1.0)


class TestObserverFactoryPlumbing:
    def _model(self) -> BottleneckResNet:
        torch.manual_seed(0)
        return BottleneckResNet(num_classes=4, bottleneck_width=4).eval()

    def test_calibrate_covers_all_policy_v1_sites(self) -> None:
        params = calibrate_activation_params(self._model(), _clean(), bits=4)
        # 8 ReLU sites + the input key.
        assert len(params) == 9

    def test_percentile_tightens_ranges_under_stress(self) -> None:
        # The intended mechanism: impulses inflate MinMax scales; the
        # percentile observer's scales stay closer to the clean ones.
        model = self._model()
        clean = _clean(n=64)
        stressed = apply_impulse_stress(clean, magnitude=10.0, seed=3)
        scale_of = lambda p: {k: float(np.asarray(v.scale)) for k, v in p.items()}  # noqa: E731
        minmax_clean = scale_of(calibrate_activation_params(model, clean, bits=4))
        minmax_stress = scale_of(calibrate_activation_params(model, stressed, bits=4))
        pct_stress = scale_of(
            calibrate_activation_params(
                model, stressed, bits=4, observer_factory=PercentileClippingObserver
            )
        )
        # Input site is where impulses land directly: MinMax must inflate.
        key = "__input__"
        assert minmax_stress[key] > minmax_clean[key] * 1.5
        assert pct_stress[key] < minmax_stress[key]

    def test_simulate_accepts_observer_factory(self) -> None:
        model = self._model()
        clean = _clean(n=48)
        sim_minmax = simulate_quantized(model, clean, SimQuantConfig(8, 4))
        sim_pct = simulate_quantized(
            model, clean, SimQuantConfig(8, 4), observer_factory=PercentileClippingObserver
        )
        x = clean.tensors[0][:8]
        with torch.no_grad():
            # Different calibration policies must yield different simulations.
            assert not torch.equal(sim_minmax(x), sim_pct(x))

    def test_default_factory_is_minmax(self) -> None:
        model = self._model()
        clean = _clean(n=48)
        a = calibrate_activation_params(model, clean, bits=8)
        b = calibrate_activation_params(model, clean, bits=8, observer_factory=MinMaxObserver)
        for k in a:
            assert float(np.asarray(a[k].scale)) == float(np.asarray(b[k].scale))
