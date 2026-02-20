"""Domain models shared across pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for '{field_name}'")
    return value.strip()


def _normalize_iso(value: str, expected_len: int, field_name: str) -> str:
    normalized = value.strip().upper()
    if len(normalized) != expected_len or not normalized.isalpha():
        raise ValueError(f"Invalid {field_name}: '{value}'")
    return normalized


@dataclass(frozen=True, slots=True)
class CountrySpec:
    """Authoritative country record from `data/un_members.yaml`."""

    iso3: str
    iso2: str
    name_en: str
    ne_admin_name: str
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CountrySpec:
        iso3 = _normalize_iso(_require_str(data.get("iso3"), "iso3"), 3, "iso3")
        iso2 = _normalize_iso(_require_str(data.get("iso2"), "iso2"), 2, "iso2")
        name_en = _require_str(data.get("name_en"), "name_en")
        ne_admin_name = _require_str(data.get("ne_admin_name"), "ne_admin_name")
        aliases_raw = data.get("aliases", [])
        aliases: list[str] = []
        if aliases_raw is not None:
            if not isinstance(aliases_raw, list):
                raise ValueError("Expected list for 'aliases'")
            for alias in aliases_raw:
                aliases.append(_require_str(alias, "aliases[]"))
        return cls(
            iso3=iso3,
            iso2=iso2,
            name_en=name_en,
            ne_admin_name=ne_admin_name,
            aliases=tuple(aliases),
        )


@dataclass(frozen=True, slots=True)
class CityRecord:
    """Candidate city loaded from Natural Earth populated places."""

    name: str
    iso3: str
    lon: float
    lat: float
    scalerank: int | None = None
    pop_max: int | None = None
    is_capital: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.iso3, self.name.casefold())


@dataclass(frozen=True, slots=True)
class CityOverride:
    """Per-country override directives."""

    n: int | None = None
    capital: str | None = None
    manual_capital: ManualCapitalOverride | None = None
    force_include: tuple[str, ...] = ()
    force_exclude: tuple[str, ...] = ()
    label_overrides: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CityOverride:
        n_raw = data.get("n")
        n: int | None
        if n_raw is None:
            n = None
        elif isinstance(n_raw, int) and n_raw >= 0:
            n = n_raw
        else:
            raise ValueError("Expected non-negative integer for override field 'n'")

        capital_raw = data.get("capital")
        capital = _require_str(capital_raw, "capital") if capital_raw is not None else None

        manual_raw = data.get("manual_capital")
        manual_capital: ManualCapitalOverride | None
        if manual_raw is None:
            manual_capital = None
        elif isinstance(manual_raw, Mapping):
            manual_capital = ManualCapitalOverride.from_mapping(manual_raw)
        else:
            raise ValueError("Expected mapping for 'manual_capital'")

        if capital is not None and manual_capital is not None:
            raise ValueError("Use only one of 'capital' or 'manual_capital' in city overrides")

        def _parse_name_list(field_name: str) -> tuple[str, ...]:
            raw = data.get(field_name, [])
            if raw is None:
                return ()
            if not isinstance(raw, list):
                raise ValueError(f"Expected list for '{field_name}'")
            return tuple(_require_str(item, f"{field_name}[]") for item in raw)

        force_include = _parse_name_list("force_include")
        force_exclude = _parse_name_list("force_exclude")

        label_raw = data.get("label_overrides", {})
        if label_raw is None:
            label_overrides: dict[str, str] = {}
        elif isinstance(label_raw, dict):
            label_overrides = {
                _require_str(k, "label_overrides key"): _require_str(v, "label_overrides value")
                for k, v in label_raw.items()
            }
        else:
            raise ValueError("Expected mapping for 'label_overrides'")

        return cls(
            n=n,
            capital=capital,
            manual_capital=manual_capital,
            force_include=force_include,
            force_exclude=force_exclude,
            label_overrides=label_overrides,
        )


@dataclass(frozen=True, slots=True)
class ManualCapitalOverride:
    """Manual capital fallback when populated places has no usable record."""

    name: str
    lon: float
    lat: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ManualCapitalOverride:
        name = _require_str(data.get("name"), "manual_capital.name")
        lon_raw = data.get("lon")
        lat_raw = data.get("lat")
        if not isinstance(lon_raw, (int, float)):
            raise ValueError("Expected numeric value for 'manual_capital.lon'")
        if not isinstance(lat_raw, (int, float)):
            raise ValueError("Expected numeric value for 'manual_capital.lat'")
        lon = float(lon_raw)
        lat = float(lat_raw)
        if lon < -180.0 or lon > 180.0:
            raise ValueError("manual_capital.lon must be between -180 and 180")
        if lat < -90.0 or lat > 90.0:
            raise ValueError("manual_capital.lat must be between -90 and 90")
        return cls(name=name, lon=lon, lat=lat)


@dataclass(frozen=True, slots=True)
class FlagOverride:
    """Manual override for Wikimedia flag lookup."""

    commons_file_title: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> FlagOverride:
        title = _require_str(data.get("commons_file_title"), "commons_file_title")
        return cls(commons_file_title=title)


@dataclass(frozen=True, slots=True)
class SelectedCities:
    """Deterministic output of city selection."""

    capital: CityRecord
    others: tuple[CityRecord, ...]

    @property
    def all_cities(self) -> tuple[CityRecord, ...]:
        return (self.capital, *self.others)


@dataclass(frozen=True, slots=True)
class BuildManifest:
    """Build metadata used for deterministic audit trails."""

    generated_at_utc: str
    config_hash_sha256: str
    git_commit: str | None
    steps: Mapping[str, str]
    artifacts: Mapping[str, str]
    skeleton_mode: bool = True

    @classmethod
    def create(
        cls,
        *,
        config_hash_sha256: str,
        git_commit: str | None,
        steps: Mapping[str, str],
        artifacts: Mapping[str, str],
    ) -> BuildManifest:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            generated_at_utc=now,
            config_hash_sha256=config_hash_sha256,
            git_commit=git_commit,
            steps=steps,
            artifacts=artifacts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "config_hash_sha256": self.config_hash_sha256,
            "git_commit": self.git_commit,
            "steps": dict(self.steps),
            "artifacts": dict(self.artifacts),
            "skeleton_mode": self.skeleton_mode,
        }


@dataclass(frozen=True, slots=True)
class AssetPaths:
    """Resolved output paths for generated media and artifacts."""

    map_png: Path
    flag_png: Path
