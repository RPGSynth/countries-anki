"""Wikimedia flag lookup and normalization boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import CountrySpec, FlagOverride


@dataclass(frozen=True, slots=True)
class FlagAttribution:
    iso3: str
    commons_title: str
    source_url: str
    license_name: str
    author: str | None
    retrieved_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "iso3": self.iso3,
            "commons_title": self.commons_title,
            "source_url": self.source_url,
            "license_name": self.license_name,
            "author": self.author,
            "retrieved_at_utc": self.retrieved_at_utc,
        }


def load_flags_overrides(path: Path) -> dict[str, FlagOverride]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")

    overrides: dict[str, FlagOverride] = {}
    for iso3_raw, value in raw.items():
        if not isinstance(iso3_raw, str):
            raise ValueError("flags override keys must be ISO3 strings")
        iso3 = iso3_raw.strip().upper()
        if len(iso3) != 3:
            raise ValueError(f"Invalid ISO3 override key: {iso3_raw}")
        if not isinstance(value, dict):
            raise ValueError(f"flags override value for {iso3} must be a mapping")
        overrides[iso3] = FlagOverride.from_mapping(value)
    return overrides


class WikimediaFlagFetcher:
    """Flag retrieval boundary.

    Full implementation (Milestone 4) should:
    - query Wikimedia Commons API deterministically
    - download SVG/PNG
    - normalize image dimensions/padding/background
    - emit attribution metadata per file
    """

    def fetch_and_normalize(
        self,
        *,
        country: CountrySpec,
        output_path: Path,
        override: FlagOverride | None = None,
    ) -> FlagAttribution:
        raise NotImplementedError(
            "Flag fetching is not implemented yet. Planned in Milestone 4."
        )
