"""Hardware-profile schema v1 and loader (ADR-014).

Every value a profile declares is an **assumption** feeding an
*estimated* analytical cost — never a measured hardware fact. The
canonical `generic_edge_npu` profile is fictional and schema v1
requires it to say so.

Precision coefficients are a LIST of entries rather than a YAML
mapping so duplicate (weight_bits, activation_bits) pairs are
detectable instead of being silently overwritten by the parser.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "HardwareProfile",
    "LoadedProfile",
    "PrecisionCost",
    "load_hardware_profile",
]

SUPPORTED_SCHEMA_VERSION = 1


class PrecisionCost(BaseModel):
    """Estimated compute cost for one (weight_bits, activation_bits) pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weight_bits: int
    activation_bits: int
    ncu: float

    @field_validator("ncu")
    @classmethod
    def _finite_nonnegative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"coefficient must be finite and >= 0, got {value}")
        return value

    @property
    def pair(self) -> tuple[int, int]:
        return (self.weight_bits, self.activation_bits)

    @property
    def label(self) -> str:
        return f"W{self.weight_bits}A{self.activation_bits}"


class HardwareProfile(BaseModel):
    """Schema-v1 analytical hardware profile (all values estimated)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    profile_name: str
    description: str
    fictional: bool
    disclaimer: str
    assumptions: list[str]
    unit: str  # definition of 1 ncu (e.g. "cost of one INT8xINT8 MAC")
    traffic_model: str  # identifier, e.g. "single-read-single-write-per-tensor-v1"
    compute_ncu_per_mac: list[PrecisionCost]
    weight_memory_ncu_per_bit: float
    activation_memory_ncu_per_bit: float
    per_layer_overhead_ncu: float
    accumulator_bits: int
    supported_weight_bits: list[int]
    supported_activation_bits: list[int]

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: int) -> int:
        if value != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value}; this loader supports "
                f"schema_version {SUPPORTED_SCHEMA_VERSION}"
            )
        return value

    @field_validator("fictional")
    @classmethod
    def _must_declare_fictional(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError(
                "schema v1 profiles must declare fictional: true — no profile in this "
                "repository represents real hardware"
            )
        return value

    @field_validator("assumptions")
    @classmethod
    def _assumptions_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("assumptions must be a non-empty list")
        return value

    @field_validator(
        "weight_memory_ncu_per_bit", "activation_memory_ncu_per_bit", "per_layer_overhead_ncu"
    )
    @classmethod
    def _coefficients_finite_nonnegative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"coefficient must be finite and >= 0, got {value}")
        return value

    @model_validator(mode="after")
    def _pairs_unique_and_supported(self) -> HardwareProfile:
        seen: set[tuple[int, int]] = set()
        for entry in self.compute_ncu_per_mac:
            if entry.pair in seen:
                raise ValueError(f"duplicate precision pair {entry.label}")
            seen.add(entry.pair)
            if entry.weight_bits not in self.supported_weight_bits:
                raise ValueError(f"{entry.label}: weight bits not in supported_weight_bits")
            if entry.activation_bits not in self.supported_activation_bits:
                raise ValueError(f"{entry.label}: activation bits not in supported_activation_bits")
        if not seen:
            raise ValueError("compute_ncu_per_mac must declare at least one precision pair")
        return self

    def compute_coefficient(self, weight_bits: int, activation_bits: int) -> float:
        """The ncu/MAC coefficient for a pair, or an actionable error."""
        for entry in self.compute_ncu_per_mac:
            if entry.pair == (weight_bits, activation_bits):
                return entry.ncu
        raise ValueError(
            f"unsupported precision pair W{weight_bits}A{activation_bits}: profile "
            f"{self.profile_name!r} declares "
            f"{sorted(e.label for e in self.compute_ncu_per_mac)}"
        )

    def canonical_digest(self) -> str:
        """SHA-256 of the canonical parsed representation (key-sorted JSON)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class LoadedProfile(BaseModel):
    """A validated profile plus its provenance hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    profile: HardwareProfile
    source_path: str
    source_sha256: str
    canonical_digest: str


def load_hardware_profile(path: str | Path) -> LoadedProfile:
    """Load + validate a schema-v1 profile; raise actionable errors.

    Legacy pre-schema files (no ``schema_version``) are rejected with a
    message naming the missing field rather than being guessed at.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"hardware profile not found: {path}")
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"hardware profile {path} is not a YAML mapping")
    if "schema_version" not in payload:
        raise ValueError(
            f"hardware profile {path} declares no schema_version: legacy/pre-schema "
            "profiles are not supported — migrate to schema v1 (ADR-014)"
        )
    profile = HardwareProfile.model_validate(payload)
    return LoadedProfile(
        profile=profile,
        source_path=str(path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_digest=profile.canonical_digest(),
    )
