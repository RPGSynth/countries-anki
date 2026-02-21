"""QA artifact generation."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Sequence

from .models import CountrySpec


def write_qa_index(
    *,
    countries: Sequence[CountrySpec],
    maps_dir: Path,
    flags_dir: Path,
    output_html: Path,
    thumbnail_width_px: int,
    max_columns: int,
    flag_status_mode: str = "strict",
) -> Path:
    """Generate an HTML index for map/flag visual QA."""
    mode = flag_status_mode.strip().casefold()
    if mode not in {"strict", "render_only"}:
        raise ValueError("flag_status_mode must be 'strict' or 'render_only'.")

    rows: list[str] = []
    for country in sorted(countries, key=lambda c: c.name_en.casefold()):
        map_name = f"map_{country.iso3}.png"
        flag_name = f"flag_{country.iso3}.png"
        map_path = maps_dir / map_name
        flag_path = flags_dir / flag_name
        map_ok = map_path.exists()
        flag_ok = flag_path.exists()

        if map_ok and flag_ok:
            card_status = "ready"
            card_status_label = "READY"
        elif map_ok:
            card_status = "map_only"
            card_status_label = "MAP_ONLY"
        else:
            card_status = "missing_map"
            card_status_label = "MISSING_MAP"

        map_src = f"../media/maps/{map_name}"
        flag_src = f"../media/flags/{flag_name}"
        if flag_ok:
            flag_status_text = "OK"
            flag_placeholder = "Flag not generated yet"
        elif mode == "render_only":
            flag_status_text = "NOT_CHECKED"
            flag_placeholder = "Flag step not run in this command"
        else:
            flag_status_text = "MISSING"
            flag_placeholder = "Flag not generated yet"
        map_cell = (
            f"  <img src='{escape(map_src)}' alt='Map {escape(country.name_en)}' "
            f"width='{thumbnail_width_px}'>"
            if map_ok
            else "  <div class='placeholder'>Map not generated</div>"
        )
        flag_cell = (
            f"  <img src='{escape(flag_src)}' alt='Flag {escape(country.name_en)}' "
            f"width='{int(thumbnail_width_px * 0.6)}'>"
            if flag_ok
            else f"  <div class='placeholder small'>{escape(flag_placeholder)}</div>"
        )

        rows.append(
            "\n".join(
                [
                    "<div class='card'>",
                    f"  <h3>{escape(country.name_en)} ({country.iso3})</h3>",
                    f"  <p class='status {card_status}'>{card_status_label}</p>",
                    f"  <p class='asset-status'>map: {'OK' if map_ok else 'MISSING'}</p>",
                    map_cell,
                    f"  <p class='asset-status'>flag: {flag_status_text}</p>",
                    flag_cell,
                    "</div>",
                ]
            )
        )

    html = "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "  <meta charset='utf-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1'>",
            "  <title>countries-anki QA</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 16px; }",
            "    .grid { "
            f"display: grid; grid-template-columns: repeat({max_columns}, minmax(220px, 1fr)); "
            "gap: 16px; }",
            "    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }",
            "    .card h3 { margin: 0 0 8px 0; font-size: 16px; }",
            "    .status { margin: 0 0 8px 0; font-weight: 700; }",
            "    .status.ready { color: #197a2f; }",
            "    .status.map_only { color: #99610f; }",
            "    .status.missing_map { color: #b22d2d; }",
            "    .asset-status { margin: 0 0 4px 0; font-size: 13px; color: #333; }",
            "    img { display: block; max-width: 100%; margin-bottom: 8px; }",
            "    .placeholder {",
            "      border: 1px dashed #bbb;",
            "      color: #666;",
            "      border-radius: 6px;",
            "      padding: 12px;",
            "      margin-bottom: 8px;",
            "      background: #fafafa;",
            "      font-size: 13px;",
            "    }",
            "    .placeholder.small { width: 60%; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Countries QA Index</h1>",
            "  <div class='grid'>",
            *rows,
            "  </div>",
            "</body>",
            "</html>",
            "",
        ]
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
