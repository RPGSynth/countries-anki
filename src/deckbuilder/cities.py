"""City override loading and deterministic city selection logic."""

from __future__ import annotations

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
        candidates: Sequence[CityRecord],
        override: CityOverride | None = None,
    ) -> SelectedCities:
        if not candidates:
            raise CitySelectionError(f"No city records available for {iso3}")

        capital = self._resolve_capital(iso3=iso3, candidates=candidates, override=override)
        n = override.n if override and override.n is not None else self.default_n
        if n < 0:
            raise CitySelectionError(f"Override n must be >= 0 for {iso3}")

        force_exclude = {name.casefold() for name in (override.force_exclude if override else ())}
        force_include = {name.casefold() for name in (override.force_include if override else ())}

        remaining = [
            city
            for city in candidates
            if city.key != capital.key and city.name.casefold() not in force_exclude
        ]
        remaining_sorted = sorted(remaining, key=_city_sort_key)

        selected: list[CityRecord] = []
        seen_keys: set[tuple[str, str]] = {capital.key}

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

        return SelectedCities(capital=capital, others=tuple(selected[:n]))

    def _resolve_capital(
        self,
        *,
        iso3: str,
        candidates: Sequence[CityRecord],
        override: CityOverride | None,
    ) -> CityRecord:
        if override and override.capital:
            target = override.capital.casefold()
            for city in candidates:
                if city.name.casefold() == target:
                    return city
            raise CitySelectionError(
                f"Override capital '{override.capital}' not found among city candidates for {iso3}"
            )

        capitals = [city for city in candidates if city.is_capital]
        if not capitals:
            raise CitySelectionError(f"No capital candidate found for {iso3}")
        return sorted(capitals, key=_capital_sort_key)[0]


def _city_sort_key(city: CityRecord) -> tuple[float, float, str]:
    scalerank = float(city.scalerank) if city.scalerank is not None else float("inf")
    pop_desc = -float(city.pop_max) if city.pop_max is not None else float("inf")
    return (scalerank, pop_desc, city.name.casefold())


def _capital_sort_key(city: CityRecord) -> tuple[float, float, str]:
    scalerank = float(city.scalerank) if city.scalerank is not None else float("inf")
    pop_desc = -float(city.pop_max) if city.pop_max is not None else float("inf")
    return (scalerank, pop_desc, city.name.casefold())
