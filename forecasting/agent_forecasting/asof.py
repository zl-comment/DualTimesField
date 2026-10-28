"""Small deterministic availability, version and same-target revision rules.

An archive modification timestamp is not a verified publication timestamp.
Likewise downloading an old notice today does not backdate first_seen_unix.
Source records carry references/hashes, not free text or future price labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Real
from urllib.parse import urlsplit

from .contracts import identifier, object_hash, sha256_string, utc_unix


def parse_utc(value: str | datetime) -> int:
    """Require an explicit UTC clock; callers must verify any local conversion."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        except ValueError as exc:
            raise ValueError("invalid explicit UTC timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp requires an explicit UTC timezone")
    if value.microsecond:
        raise ValueError("timestamp must resolve to whole UTC seconds")
    return utc_unix(int(value.astimezone(timezone.utc).timestamp()))


@dataclass(frozen=True, slots=True)
class SourceVersion:
    """One explicit version of a source's evidence.

    A normal replacement wholly supersedes its predecessor. A withdrawal
    revokes the chain from ``valid_from_unix`` onward; ``valid_to_unix`` does
    not reinstate old evidence. Before that withdrawal takes effect, its
    explicit predecessor still governs. These are evidence availability
    rules, not inferred restoration of capacity or any numerical variable.
    """
    chain_id: str
    version_id: str
    source_url: str
    source_sha256: str
    published_unix: int | None
    first_seen_unix: int | None
    valid_from_unix: int
    valid_to_unix: int
    publication_basis: str
    supersedes_id: str | None = None
    withdrawn: bool = False

    def __post_init__(self) -> None:
        identifier(self.chain_id, "chain_id")
        identifier(self.version_id, "version_id")
        sha256_string(self.source_sha256, "source_sha256")
        if not isinstance(self.source_url, str):
            raise ValueError("source_url must be an HTTP(S) reference")
        url = urlsplit(self.source_url)
        if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password:
            raise ValueError("source_url must be a public HTTP(S) reference")
        if self.publication_basis not in {"verified", "archive_proxy", "unknown"}:
            raise ValueError("unknown publication clock basis")
        for name in ("published_unix", "first_seen_unix", "valid_from_unix", "valid_to_unix"):
            value = getattr(self, name)
            if value is None and name in {"published_unix", "first_seen_unix"}:
                continue
            object.__setattr__(self, name, utc_unix(value, name))
        if self.valid_to_unix <= self.valid_from_unix:
            raise ValueError("effective interval must be nonempty [from, to)")
        if self.publication_basis == "verified" and self.published_unix is None:
            raise ValueError("verified publication needs a timestamp")
        if (self.published_unix is not None and self.first_seen_unix is not None
                and self.first_seen_unix < self.published_unix):
            raise ValueError("first_seen cannot precede public issuance")
        if self.supersedes_id is not None:
            identifier(self.supersedes_id, "supersedes_id")
            if self.supersedes_id == self.version_id:
                raise ValueError("a version cannot supersede itself")
        if not isinstance(self.withdrawn, bool):
            raise ValueError("withdrawn must be boolean")

    def visible_at(self, origin_unix: int) -> bool:
        origin = utc_unix(origin_unix)
        return (self.publication_basis == "verified" and self.published_unix is not None
                and self.first_seen_unix is not None
                and self.published_unix <= origin and self.first_seen_unix <= origin)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class AsOfSnapshot:
    origin_unix: int
    visible_versions: tuple[SourceVersion, ...]
    selected_versions: tuple[SourceVersion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_unix", utc_unix(self.origin_unix))
        object.__setattr__(self, "visible_versions", tuple(self.visible_versions))
        object.__setattr__(self, "selected_versions", tuple(self.selected_versions))
        if any(not isinstance(v, SourceVersion) or not v.visible_at(self.origin_unix)
               for v in self.visible_versions):
            raise ValueError("snapshot includes unavailable source versions")
        object.__setattr__(self, "visible_versions", tuple(sorted(set(self.visible_versions),
                           key=lambda v: (v.chain_id, v.published_unix, v.first_seen_unix, v.version_id))))
        expected = _select_versions(self.visible_versions)
        if self.selected_versions != expected:
            raise ValueError("snapshot selection does not follow the version chain")

    @property
    def snapshot_id(self) -> str:
        return object_hash({"origin_unix": self.origin_unix,
                            "visible_versions": [v.to_dict() for v in self.visible_versions]})

    def active_for(self, target_time_unix: int) -> tuple[SourceVersion, ...]:
        """Resolve a future-effective withdrawal separately from its receipt.

        Only withdrawals permit predecessor lookup before their effective
        start. An ordinary replacement, including a postponed event, remains
        a whole-version replacement even before its new effective interval.
        """
        target = utc_unix(target_time_unix)
        if not self.origin_unix <= target < self.origin_unix + 24 * 3600:
            raise ValueError("target is outside this snapshot's 24-hour horizon")
        by_id = {(v.chain_id, v.version_id): v for v in self.visible_versions}
        active = []
        for selected in self.selected_versions:
            version = selected
            while version.withdrawn:
                if target >= version.valid_from_unix or version.supersedes_id is None:
                    version = None
                    break
                # Snapshot construction has already checked a complete,
                # acyclic, single-predecessor chain using visible records.
                version = by_id[(version.chain_id, version.supersedes_id)]
            if version is not None and version.valid_from_unix <= target < version.valid_to_unix:
                active.append(version)
        return tuple(active)


def _select_versions(visible: tuple[SourceVersion, ...]) -> tuple[SourceVersion, ...]:
    identities = {}
    chains = {}
    for version in visible:
        identity = (version.chain_id, version.version_id)
        if identity in identities and identities[identity] != version:
            raise ValueError("conflicting content for one source version")
        identities[identity] = version
    for version in identities.values():
        chains.setdefault(version.chain_id, []).append(version)
    selected = []
    for chain_id in sorted(chains):
        # Version identifiers are opaque. Explicit relationships resolve even
        # equal-second issuance without guessing lexical/numeric ID ordering.
        chain = {v.version_id: v for v in chains[chain_id]}
        roots = [v for v in chain.values() if v.supersedes_id is None]
        if len(roots) != 1:
            raise ValueError("incomplete or conflicting source version chain")
        children = {}
        for version in chain.values():
            parent = version.supersedes_id
            if parent is None:
                continue
            if parent not in chain or parent in children:
                raise ValueError("incomplete or conflicting source version chain")
            if version.published_unix < chain[parent].published_unix:
                raise ValueError("revision publication precedes its predecessor")
            children[parent] = version
        current, visited = roots[0], set()
        while current.version_id not in visited:
            visited.add(current.version_id)
            if current.version_id not in children:
                break
            current = children[current.version_id]
        if len(visited) != len(chain):
            raise ValueError("cyclic or disconnected source version chain")
        selected.append(current)
    return tuple(selected)


def build_snapshot(origin_unix: int, versions) -> AsOfSnapshot:
    origin = utc_unix(origin_unix)
    records = tuple(versions)
    if any(not isinstance(v, SourceVersion) for v in records):
        raise ValueError("snapshot accepts only validated SourceVersion records")
    visible = tuple(sorted(set(v for v in records if v.visible_at(origin)),
                           key=lambda v: (v.chain_id, v.published_unix, v.first_seen_unix, v.version_id)))
    return AsOfSnapshot(origin, visible, _select_versions(visible))


@dataclass(frozen=True, slots=True)
class ForecastVintage:
    source_id: str
    vintage_id: str
    variable: str
    region: str
    unit: str
    target_time_unix: int
    issued_unix: int
    first_seen_unix: int
    value: float
    covered: bool = True

    def __post_init__(self) -> None:
        for key in ("source_id", "vintage_id", "variable", "region", "unit"):
            identifier(getattr(self, key), key)
        for key in ("target_time_unix", "issued_unix", "first_seen_unix"):
            object.__setattr__(self, key, utc_unix(getattr(self, key), key))
        if self.first_seen_unix < self.issued_unix:
            raise ValueError("forecast was observed before its issuance")
        if not isinstance(self.covered, bool):
            raise ValueError("covered must be boolean")
        if isinstance(self.value, bool) or not isinstance(self.value, Real) or not math.isfinite(self.value):
            raise ValueError("forecast value must be finite")
        object.__setattr__(self, "value", float(self.value))


def same_target_revision(older: ForecastVintage, newer: ForecastVintage, origin_unix: int) -> float:
    """Difference of two *covered*, available forecasts for one delivery time.

    This does not create a past covariate, infer price effects, or treat repeated
    source runs as new information.  No imputed uncovered tail is accepted.
    """
    origin = utc_unix(origin_unix)
    for field in ("source_id", "variable", "region", "unit", "target_time_unix"):
        if getattr(older, field) != getattr(newer, field):
            raise ValueError("revision requires identical target and field semantics")
    for record in (older, newer):
        if not record.covered or max(record.issued_unix, record.first_seen_unix) > origin:
            raise ValueError("revision uses unavailable or uncovered forecasts")
    if newer.vintage_id == older.vintage_id or (newer.issued_unix, newer.first_seen_unix) <= (older.issued_unix, older.first_seen_unix):
        raise ValueError("revision requires a strictly newer, distinct vintage")
    if not origin <= newer.target_time_unix < origin + 24 * 3600:
        raise ValueError("revision target is outside the forecast horizon")
    return newer.value - older.value
