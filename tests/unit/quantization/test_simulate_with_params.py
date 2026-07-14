"""Unit tests for simulation from precomputed activation parameters
(the ADR-012 mechanism-decomposition entry point)."""

import pytest
import torch

from quantscope.data.texture10 import Texture10Params, make_texture10
from quantscope.models.bottleneck_resnet import BottleneckResNet
from quantscope.quantization.simulate import (
    SimQuantConfig,
    calibrate_activation_params,
    quantize_weights_uniform,
    simulate_quantized,
    simulate_quantized_with_params,
)

_SMALL = Texture10Params(num_classes=4, image_size=16)


def _model() -> BottleneckResNet:
    torch.manual_seed(0)
    return BottleneckResNet(num_classes=4, bottleneck_width=4).eval()


def _calib(n: int = 48):
    return make_texture10(num_samples=n, seed=0, params=_SMALL)


class TestSimulateQuantizedWithParams:
    def test_matches_simulate_quantized_for_unmixed_params(self) -> None:
        # Policy v1 calibrates on the weight-quantized model; parameters
        # obtained that way and re-attached must reproduce
        # simulate_quantized exactly.
        model = _model()
        calib = _calib()
        reference = simulate_quantized(model, calib, SimQuantConfig(4, 4))
        weight_quantized = quantize_weights_uniform(model, bits=4)
        params = calibrate_activation_params(weight_quantized, calib, bits=4)
        rebuilt = simulate_quantized_with_params(model, params, weight_bits=4)
        x = calib.tensors[0][:8]
        with torch.no_grad():
            torch.testing.assert_close(rebuilt(x), reference(x), rtol=0.0, atol=0.0)

    def test_substituted_site_changes_output(self) -> None:
        model = _model()
        calib = _calib()
        params = calibrate_activation_params(model, calib, bits=4)
        doubled = dict(params)
        original = params["__input__"]
        doubled["__input__"] = type(original)(
            scale=original.scale * 2,
            zero_point=original.zero_point,
            qmin=original.qmin,
            qmax=original.qmax,
            bits=original.bits,
            signed=original.signed,
            scheme=original.scheme,
            granularity=original.granularity,
            channel_axis=original.channel_axis,
        )
        base = simulate_quantized_with_params(model, params, weight_bits=4)
        mixed = simulate_quantized_with_params(model, doubled, weight_bits=4)
        x = calib.tensors[0][:8]
        with torch.no_grad():
            assert not torch.equal(base(x), mixed(x))

    def test_incomplete_site_coverage_rejected(self) -> None:
        model = _model()
        params = calibrate_activation_params(model, _calib(), bits=4)
        missing = {k: v for k, v in params.items() if k != "stem_relu"}
        with pytest.raises(ValueError, match="stem_relu"):
            simulate_quantized_with_params(model, missing, weight_bits=4)
        extra = {**params, "not_a_site": params["__input__"]}
        with pytest.raises(ValueError, match="not_a_site"):
            simulate_quantized_with_params(model, extra, weight_bits=4)

    def test_original_model_untouched(self) -> None:
        model = _model()
        params = calibrate_activation_params(model, _calib(), bits=4)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        simulate_quantized_with_params(model, params, weight_bits=4)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
