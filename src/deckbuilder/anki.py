"""Anki deck packaging boundary using genanki."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import AnkiConfig, ProjectConfig
from .models import CountrySpec


@dataclass(frozen=True, slots=True)
class NotePayload:
    country: CountrySpec
    map_filename: str
    flag_filename: str


class AnkiDeckBuilder:
    """Build `.apkg` via genanki.

    Full implementation (Milestone 5) will instantiate:
    - a stable deck id + model id
    - templates/CSS from config
    - one note per country with deterministic media names
    """

    def __init__(self, project_cfg: ProjectConfig, anki_cfg: AnkiConfig) -> None:
        self.project_cfg = project_cfg
        self.anki_cfg = anki_cfg

    def build_deck(self, *, notes: Sequence[NotePayload], media_files: Sequence[Path], output: Path) -> Path:
        raise NotImplementedError("Anki deck packaging is not implemented yet. Planned in Milestone 5.")
