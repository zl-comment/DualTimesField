"""Immutable input boundary; no evaluation labels or arbitrary metadata.

This is the single-origin counterpart of ``common.foundation_api.PredictionView``.
It deliberately does not accept an AlignedSplit, NPZ, labels, or an opaque context
object.  Tuples and copied read-only mappings also close PredictionView's shallow
immutability gap.  Validation cannot establish a data supplier's truthfulness:
the loader must separately prove the provenance and availability of each value.

The ``future_forecasts`` mapping is the candidate-2 typed future-forecast
extension: a known-future forecast tensor (e.g. predispatch TOTALDEMAND)
that legitimately has NO same-semantic past observations.  It is exempt
from the calendar rule that future covariates need same-named past values;
inventing a past history to satisfy that rule is forbidden, and the frozen
tool instead encodes the honest absence itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping


CALENDAR_NAMES = tuple(f"cal_{index}" for index in range(10))
PAST_COVARIATE_NAMES = frozenset(("demand_std", *CALENDAR_NAMES))
FUTURE_COVARIATE_NAMES = frozenset(CALENDAR_NAMES)
TYPED_FUTURE_FORECAST_NAMES = frozenset(("predispatchis_totaldemand_mw",))
METADATA_NAMES = frozenset({"region", "snapshot_id", "input_profile",
                            "source_manifest_sha256", "transform_sha256"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def identifier(value: str, name: str = "identifier") -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a bounded plain identifier")
    return value


def sha256_string(value: str, name: str = "sha256") -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def utc_unix(value: int, name: str = "time") -> int:
    """Accept integral UTC Unix *seconds*, never naive/local time or booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be integral UTC Unix seconds")
    result = int(value)
    try:
        datetime.fromtimestamp(result, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{name} is outside supported Unix seconds") from exc
    return result


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def object_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite_tuple(values, length: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a numeric sequence")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    if len(raw) != length:
        raise ValueError(f"{name} must have length {length}")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in raw):
        raise ValueError(f"{name} must contain real numbers")
    result = tuple(float(v) for v in raw)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True, slots=True)
class FrozenForecastInput:
    origin_unix: int
    target_time_unix: tuple[int, ...]
    price_history: tuple[float, ...]
    past_covariates: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    future_covariates: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    future_forecasts: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        origin = utc_unix(self.origin_unix, "origin_unix")
        if origin % 3600:
            raise ValueError("origin must be aligned to a UTC hour")
        targets = tuple(utc_unix(v, "target_time_unix") for v in self.target_time_unix)
        if targets != tuple(origin + 3600 * h for h in range(24)):
            raise ValueError("targets must equal origin + [0..23] hours")
        price = _finite_tuple(self.price_history, 72, "price_history")
        mappings = []
        for name, values, allowed, length in (
            ("past_covariates", self.past_covariates, PAST_COVARIATE_NAMES, 72),
            ("future_covariates", self.future_covariates, FUTURE_COVARIATE_NAMES, 24),
        ):
            if not isinstance(values, Mapping) or not set(values).issubset(allowed):
                raise ValueError(f"{name} contains unapproved fields")
            mappings.append({key: _finite_tuple(values[key], length, f"{name}.{key}")
                             for key in sorted(values)})
        past, future = mappings
        if not set(future).issubset(past):
            raise ValueError("future covariates need same-semantic past covariates")
        if not isinstance(self.future_forecasts, Mapping) or not set(self.future_forecasts).issubset(TYPED_FUTURE_FORECAST_NAMES):
            raise ValueError("future_forecasts contains unapproved fields")
        forecasts = {key: _finite_tuple(self.future_forecasts[key], 24, f"future_forecasts.{key}")
                     for key in sorted(self.future_forecasts)}
        if not isinstance(self.metadata, Mapping) or not set(self.metadata).issubset(METADATA_NAMES):
            raise ValueError("metadata contains unapproved fields")
        meta = {}
        for key in sorted(self.metadata):
            value = self.metadata[key]
            if key.endswith("sha256"):
                meta[key] = sha256_string(value, key)
            else:
                meta[key] = identifier(value, key)
        if "region" in meta and meta["region"] not in {"NSW1", "QLD1", "TAS1", "VIC1", "SA1"}:
            raise ValueError("unsupported NEM region")
        object.__setattr__(self, "origin_unix", origin)
        object.__setattr__(self, "target_time_unix", targets)
        object.__setattr__(self, "price_history", price)
        object.__setattr__(self, "past_covariates", MappingProxyType(past))
        object.__setattr__(self, "future_covariates", MappingProxyType(future))
        object.__setattr__(self, "future_forecasts", MappingProxyType(forecasts))
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    def to_dict(self) -> dict:
        """Return independent JSON-compatible data, never a mutable backdoor."""
        return {"origin_unix": self.origin_unix,
                "target_time_unix": list(self.target_time_unix),
                "price_history": list(self.price_history),
                "past_covariates": {k: list(v) for k, v in self.past_covariates.items()},
                "future_covariates": {k: list(v) for k, v in self.future_covariates.items()},
                "future_forecasts": {k: list(v) for k, v in self.future_forecasts.items()},
                "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "FrozenForecastInput":
        required = {"origin_unix", "target_time_unix", "price_history"}
        allowed = required | {"past_covariates", "future_covariates",
                              "future_forecasts", "metadata"}
        if not isinstance(payload, Mapping) or not required.issubset(payload) or not set(payload).issubset(allowed):
            raise ValueError("forecast input has missing or unapproved fields")
        return cls(**dict(payload))

    def input_hash(self) -> str:
        return object_hash(self.to_dict())
