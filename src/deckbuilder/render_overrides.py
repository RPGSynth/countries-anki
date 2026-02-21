"""Render override loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import RenderOverride


def load_render_overrides(path: Path) -> dict[str, RenderOverride]:
    """Load optional per-country render overrides keyed by ISO3."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")

    overrides: dict[str, RenderOverride] = {}
    for iso3_raw, value in raw.items():
        if not isinstance(iso3_raw, str):
            raise ValueError(f"Render override key must be ISO3 string in {path}")
        iso3 = iso3_raw.strip().upper()
        if len(iso3) != 3:
            raise ValueError(f"Invalid ISO3 override key '{iso3_raw}' in {path}")
        if not isinstance(value, dict):
            raise ValueError(f"Render override value for {iso3} must be a mapping in {path}")
        overrides[iso3] = RenderOverride.from_mapping(value)
    return overrides
