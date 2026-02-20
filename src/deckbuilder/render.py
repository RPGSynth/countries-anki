"""Map rendering boundary for country maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RenderConfig
from .models import SelectedCities


@dataclass(frozen=True, slots=True)
class RenderRequest:
    iso3: str
    geometry: Any
    selected_cities: SelectedCities
    output_path: Path


class MapRenderer:
    """Country map renderer.

    The final implementation will include:
    - deterministic extent/padding/projection
    - marker and label collision handling (capital never dropped)
    - stable image output settings
    """

    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg

    def render(self, req: RenderRequest) -> Path:
        raise NotImplementedError(
            "Map rendering is not implemented yet. "
            "Planned in Milestone 3 (render outlines + cities + deterministic labels)."
        )
