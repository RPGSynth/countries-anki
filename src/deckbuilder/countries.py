"""UN member country list loading and indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from .models import CountrySpec


def load_un_members(path: Path) -> list[CountrySpec]:
    """Load and validate the authoritative UN-member country list."""
    if not path.exists():
        raise FileNotFoundError(f"UN members file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}")

    countries: list[CountrySpec] = []
    seen_iso3: set[str] = set()
    seen_iso2: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Expected mapping at index {idx} in {path}")
        country = CountrySpec.from_mapping(item)
        if country.iso3 in seen_iso3:
            raise ValueError(f"Duplicate ISO3 '{country.iso3}' in {path}")
        if country.iso2 in seen_iso2:
            raise ValueError(f"Duplicate ISO2 '{country.iso2}' in {path}")
        seen_iso3.add(country.iso3)
        seen_iso2.add(country.iso2)
        countries.append(country)
    return countries


def country_index_by_iso3(countries: Iterable[CountrySpec]) -> dict[str, CountrySpec]:
    return {country.iso3: country for country in countries}


def country_name_candidates(country: CountrySpec) -> tuple[str, ...]:
    """Return possible names that can be used for joins/fuzzy matching."""
    return (country.name_en, country.ne_admin_name, country.capital, *country.aliases)
