"""Anki deck packaging using genanki."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Sequence

from .config import AppConfig, AnkiConfig, ProjectConfig
from .countries import load_un_members
from .models import CountrySpec


@dataclass(frozen=True, slots=True)
class NotePayload:
    country: CountrySpec
    map_filename: str
    flag_filename: str


@dataclass(slots=True)
class DeckBuildReport:
    output_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.infos.append(msg)


class AnkiDeckBuilder:
    """Build a deterministic `.apkg` deck via genanki."""

    def __init__(self, project_cfg: ProjectConfig, anki_cfg: AnkiConfig) -> None:
        self.project_cfg = project_cfg
        self.anki_cfg = anki_cfg

    def build_deck(
        self,
        *,
        notes: Sequence[NotePayload],
        media_files: Sequence[Path],
        output: Path,
    ) -> Path:
        genanki = _require_genanki()
        model = self._build_model(genanki)
        deck = genanki.Deck(self.project_cfg.deck_id, self.project_cfg.deck_name)

        note_tags = [tag for tag in self.anki_cfg.tags if tag]
        for payload in notes:
            map_html = _image_html(
                filename=payload.map_filename,
                css_class="map-image",
                alt=f"Map of {payload.country.name_en}",
            )
            flag_html = _image_html(
                filename=payload.flag_filename,
                css_class="flag-image",
                alt=f"Flag of {payload.country.name_en}",
            )
            note = genanki.Note(
                model=model,
                fields=[
                    payload.country.name_en,
                    map_html,
                    flag_html,
                    payload.country.iso3,
                    " ".join(note_tags),
                ],
                guid=genanki.guid_for(payload.country.iso3),
                tags=note_tags,
            )
            deck.add_note(note)

        output.parent.mkdir(parents=True, exist_ok=True)
        package = genanki.Package(deck)
        package.media_files = [str(path) for path in media_files]
        package.write_to_file(str(output))
        return output

    def _build_model(self, genanki: Any) -> Any:
        model_name = f"{self.project_cfg.deck_name} Model"
        return genanki.Model(
            self.project_cfg.model_id,
            model_name,
            fields=[
                {"name": "CountryName"},
                {"name": "MapImage"},
                {"name": "FlagImage"},
                {"name": "ISO3"},
                {"name": "Tags"},
            ],
            templates=[
                {
                    "name": "FlagToMap",
                    "qfmt": _front_template(self.anki_cfg),
                    "afmt": _back_template(self.anki_cfg),
                }
            ],
            css=_card_css(self.anki_cfg),
        )


def run_build_deck(cfg: AppConfig) -> DeckBuildReport:
    report = DeckBuildReport(output_path=cfg.project.output_apkg)
    countries = sorted(load_un_members(cfg.paths.un_members), key=lambda item: item.iso3)
    report.add_info(f"Loaded {len(countries)} UN member records from {cfg.paths.un_members}")
    if not countries:
        report.add_error("UN members list is empty; cannot build Anki deck.")
        return report

    notes, media_files, missing_maps, missing_flags = _collect_notes_and_media(
        countries=countries,
        maps_dir=cfg.paths.maps_dir,
        flags_dir=cfg.paths.flags_dir,
    )
    if missing_maps:
        report.add_error(
            "Missing rendered maps for deck build: " + _format_code_list(sorted(missing_maps))
        )
    if missing_flags:
        report.add_error(
            "Missing flag images for deck build: " + _format_code_list(sorted(missing_flags))
        )
    if report.errors:
        return report

    builder = AnkiDeckBuilder(cfg.project, cfg.anki)
    try:
        output_path = builder.build_deck(
            notes=notes,
            media_files=media_files,
            output=cfg.project.output_apkg,
        )
    except Exception as exc:
        report.add_error(f"Deck packaging failed: {exc}")
        return report

    report.output_path = output_path
    report.summary = {
        "countries_total": len(countries),
        "notes_written": len(notes),
        "media_files": len(media_files),
    }
    report.add_info(
        "Deck summary: "
        f"countries_total={len(countries)}, "
        f"notes_written={len(notes)}, "
        f"media_files={len(media_files)}"
    )
    report.add_info(f"Deck written to {output_path}")
    return report


def format_deck_lines(report: DeckBuildReport) -> Sequence[str]:
    lines: list[str] = []
    lines.extend(f"[INFO] {msg}" for msg in report.infos)
    lines.extend(f"[WARN] {msg}" for msg in report.warnings)
    lines.extend(f"[ERROR] {msg}" for msg in report.errors)
    if report.ok:
        lines.append("[OK] Deck packaging completed with no errors.")
    return lines


def _collect_notes_and_media(
    *,
    countries: Sequence[CountrySpec],
    maps_dir: Path,
    flags_dir: Path,
) -> tuple[list[NotePayload], list[Path], list[str], list[str]]:
    notes: list[NotePayload] = []
    media: list[Path] = []
    missing_maps: list[str] = []
    missing_flags: list[str] = []
    for country in countries:
        map_filename = f"map_{country.iso3}.png"
        flag_filename = f"flag_{country.iso3}.png"
        map_path = maps_dir / map_filename
        flag_path = flags_dir / flag_filename

        has_map = map_path.exists()
        has_flag = flag_path.exists()
        if not has_map:
            missing_maps.append(country.iso3)
        if not has_flag:
            missing_flags.append(country.iso3)
        if not has_map or not has_flag:
            continue

        notes.append(
            NotePayload(
                country=country,
                map_filename=map_filename,
                flag_filename=flag_filename,
            )
        )
        media.append(map_path)
        media.append(flag_path)
    return (notes, media, missing_maps, missing_flags)


def _image_html(*, filename: str, css_class: str, alt: str) -> str:
    safe_filename = escape(filename, quote=True)
    safe_alt = escape(alt, quote=True)
    return f"<img src='{safe_filename}' class='{css_class}' alt='{safe_alt}'>"


def _front_template(cfg: AnkiConfig) -> str:
    _ = cfg
    return "<div class='flag-wrap'>{{FlagImage}}</div>"


def _back_template(cfg: AnkiConfig) -> str:
    title_size = cfg.templates.back_title_size_px
    return (
        "<div class='country-title back-title' "
        f"style='font-size:{title_size}px;'>{{{{CountryName}}}}</div>\n"
        "<div class='map-wrap'>{{MapImage}}</div>"
    )


def _card_css(cfg: AnkiConfig) -> str:
    map_max_width = cfg.templates.map_max_width_pct
    flag_max_height = cfg.templates.flag_max_height_px
    return "\n".join(
        [
            ".card { text-align: center; }",
            ".country-title { font-weight: 700; margin: 0 0 12px 0; line-height: 1.2; }",
            ".map-wrap, .flag-wrap { margin: 0 auto; }",
            ".map-image { display: block; "
            f"max-width: {map_max_width}%; margin: 0 auto; height: auto; }}",
            f".flag-image {{ display: block; max-height: {flag_max_height}px; max-width: 100%; "
            "margin: 0 auto; height: auto; }}",
        ]
    )


def _require_genanki() -> Any:
    try:
        import genanki
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("genanki is required for deck packaging") from exc
    return genanki


def _format_code_list(values: list[str], limit: int = 12) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    shown = ", ".join(values[:limit])
    return f"{shown}, ... (+{len(values) - limit} more)"
