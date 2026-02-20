"""City override loading and deterministic city selection logic."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Sequence

import yaml

from .models import CityOverride, CityRecord, SelectedCities


class CitySelectionError(ValueError):
    """Raised when capital/city selection fails for a country."""


def load_cities_overrides(path: Path) -> dict[str, CityOverride]:
    """Load optional per-country city selection overrides."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")

    overrides: dict[str, CityOverride] = {}
    for iso3_raw, value in raw.items():
        if not isinstance(iso3_raw, str):
            raise ValueError(f"Override key must be ISO3 string in {path}")
        iso3 = iso3_raw.strip().upper()
        if len(iso3) != 3:
            raise ValueError(f"Invalid ISO3 override key '{iso3_raw}' in {path}")
        if not isinstance(value, dict):
            raise ValueError(f"Override value for {iso3} must be a mapping in {path}")
        overrides[iso3] = CityOverride.from_mapping(value)
    return overrides


class CitySelector:
    """Deterministic selection: capital + top N additional cities."""

    def __init__(self, default_n: int) -> None:
        if default_n < 0:
            raise ValueError("default_n must be >= 0")
        self.default_n = default_n

    def select_for_country(
        self,
        *,
        iso3: str,
        canonical_capital: str,
        candidates: Sequence[CityRecord],
        override: CityOverride | None = None,
        target_n: int | None = None,
    ) -> SelectedCities:
        if not candidates and not (override and override.manual_capital):
            raise CitySelectionError(f"No city records available for {iso3}")

        capital, capital_source_key, capital_resolution = self._resolve_capital(
            iso3=iso3,
            canonical_capital=canonical_capital,
            candidates=candidates,
            override=override,
        )
        if target_n is not None:
            n = target_n
        elif override and override.n is not None:
            n = override.n
        else:
            n = self.default_n
        if n < 0:
            raise CitySelectionError(f"Override n must be >= 0 for {iso3}")

        force_exclude = {name.casefold() for name in (override.force_exclude if override else ())}
        force_include = {name.casefold() for name in (override.force_include if override else ())}

        remaining = [
            city
            for city in candidates
            if (capital_source_key is None or city.key != capital_source_key)
            and city.name.casefold() not in force_exclude
        ]
        remaining_sorted = sorted(remaining, key=_city_sort_key)

        selected: list[CityRecord] = []
        seen_keys: set[tuple[str, str]]
        if capital_source_key is None:
            seen_keys = {capital.key}
        else:
            seen_keys = {capital_source_key}

        for city in remaining_sorted:
            if city.name.casefold() in force_include and city.key not in seen_keys:
                selected.append(city)
                seen_keys.add(city.key)
            if len(selected) >= n:
                break

        if len(selected) < n:
            for city in remaining_sorted:
                if city.key in seen_keys:
                    continue
                selected.append(city)
                seen_keys.add(city.key)
                if len(selected) >= n:
                    break

        return SelectedCities(
            capital=capital,
            others=tuple(selected[:n]),
            effective_n=n,
            capital_resolution=capital_resolution,
        )

    def _resolve_capital(
        self,
        *,
        iso3: str,
        canonical_capital: str,
        candidates: Sequence[CityRecord],
        override: CityOverride | None,
    ) -> tuple[CityRecord, tuple[str, str] | None, str]:
        targets: list[tuple[str, str]] = [("canonical_name_match", canonical_capital)]
        if override and override.capital:
            if not names_match(override.capital, canonical_capital):
                targets.append(("override_name_match", override.capital))

        for resolution, target in targets:
            matches = [city for city in candidates if names_match(city.name, target)]
            if matches:
                city = sorted(matches, key=_capital_sort_key)[0]
                return (city, city.key, resolution)

        if override and override.manual_capital:
            manual = override.manual_capital
            return (
                CityRecord(
                    name=manual.name,
                    iso3=iso3,
                    lon=manual.lon,
                    lat=manual.lat,
                    scalerank=0,
                    pop_max=None,
                    is_capital=True,
                ),
                None,
                "manual_override",
            )

        if candidates:
            suspected_pool = [city for city in candidates if city.is_capital]
            if not suspected_pool:
                suspected_pool = list(candidates)
            suspected = sorted(suspected_pool, key=_capital_sort_key)[0]
            relabeled = CityRecord(
                name=canonical_capital,
                iso3=iso3,
                lon=suspected.lon,
                lat=suspected.lat,
                scalerank=suspected.scalerank,
                pop_max=suspected.pop_max,
                is_capital=True,
            )
            return (relabeled, suspected.key, "auto_relabel")

        raise CitySelectionError(
            f"Capital '{canonical_capital}' from un_members.yaml was not found in "
            f"city candidates for {iso3}"
        )


def _city_sort_key(city: CityRecord) -> tuple[float, float, str]:
    scalerank = float(city.scalerank) if city.scalerank is not None else float("inf")
    pop_desc = -float(city.pop_max) if city.pop_max is not None else float("inf")
    return (scalerank, pop_desc, city.name.casefold())


def _capital_sort_key(city: CityRecord) -> tuple[float, float, str]:
    scalerank = float(city.scalerank) if city.scalerank is not None else float("inf")
    pop_desc = -float(city.pop_max) if city.pop_max is not None else float("inf")
    return (scalerank, pop_desc, city.name.casefold())


def names_match(left: str, right: str) -> bool:
    return _normalize_name(left) == _normalize_name(right)


def _normalize_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return "".join(ch for ch in without_marks.casefold() if ch.isalnum())
