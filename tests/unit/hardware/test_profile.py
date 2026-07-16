"""Schema-v1 hardware-profile validation tests (ADR-014)."""

from pathlib import Path

import pytest
import yaml

from quantscope.hardware import load_hardware_profile

CANONICAL = Path("configs/hardware/generic_edge_npu.yaml")
LEGACY = Path("tests/fixtures/hardware/generic_edge_npu_legacy_v0.yaml")


def _payload() -> dict:
    return yaml.safe_load(CANONICAL.read_text())


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


class TestCanonicalProfile:
    def test_loads_and_hashes(self) -> None:
        loaded = load_hardware_profile(CANONICAL)
        assert loaded.profile.profile_name == "generic_edge_npu"
        assert loaded.profile.fictional is True
        assert len(loaded.source_sha256) == 64
        assert len(loaded.canonical_digest) == 64

    def test_digest_stable_across_loads(self) -> None:
        assert (
            load_hardware_profile(CANONICAL).canonical_digest
            == load_hardware_profile(CANONICAL).canonical_digest
        )

    def test_digest_changes_with_coefficients(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["compute_ncu_per_mac"][1]["ncu"] = 0.6
        changed = load_hardware_profile(_write(tmp_path, payload))
        assert changed.canonical_digest != load_hardware_profile(CANONICAL).canonical_digest

    def test_coefficient_lookup(self) -> None:
        profile = load_hardware_profile(CANONICAL).profile
        assert profile.compute_coefficient(8, 8) == 1.0
        assert profile.compute_coefficient(4, 4) == 0.55
        with pytest.raises(ValueError, match="W2A2"):
            profile.compute_coefficient(2, 2)


class TestLegacyProfileRejected:
    def test_legacy_v0_fixture_fails_with_schema_error(self) -> None:
        # The pre-schema throughput/bandwidth format is preserved
        # verbatim and must be rejected, not guessed at.
        with pytest.raises(ValueError, match="schema_version"):
            load_hardware_profile(LEGACY)


class TestRejectionRules:
    def test_unsupported_schema_version(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["schema_version"] = 2
        with pytest.raises(ValueError, match="unsupported schema_version"):
            load_hardware_profile(_write(tmp_path, payload))

    def test_unknown_field_forbidden(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["dram_bandwidth_gb_per_s"] = 8.0
        with pytest.raises(ValueError, match="dram_bandwidth"):
            load_hardware_profile(_write(tmp_path, payload))

    def test_duplicate_pair_rejected(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["compute_ncu_per_mac"].append({"weight_bits": 8, "activation_bits": 8, "ncu": 0.9})
        with pytest.raises(ValueError, match="duplicate precision pair W8A8"):
            load_hardware_profile(_write(tmp_path, payload))

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_bad_coefficients_rejected(self, tmp_path: Path, bad: float) -> None:
        payload = _payload()
        payload["weight_memory_ncu_per_bit"] = bad
        with pytest.raises(ValueError, match="finite"):
            load_hardware_profile(_write(tmp_path, payload))

    def test_pair_outside_supported_bits(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["compute_ncu_per_mac"].append({"weight_bits": 2, "activation_bits": 8, "ncu": 0.4})
        with pytest.raises(ValueError, match="supported_weight_bits"):
            load_hardware_profile(_write(tmp_path, payload))

    def test_fictional_false_rejected(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["fictional"] = False
        with pytest.raises(ValueError, match="fictional"):
            load_hardware_profile(_write(tmp_path, payload))

    def test_empty_assumptions_rejected(self, tmp_path: Path) -> None:
        payload = _payload()
        payload["assumptions"] = []
        with pytest.raises(ValueError, match="assumptions"):
            load_hardware_profile(_write(tmp_path, payload))

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_hardware_profile(tmp_path / "absent.yaml")
