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
) -> Path:
    """Generate a simple HTML index for visual QA."""
    rows: list[str] = []
    for country in sorted(countries, key=lambda c: c.name_en.casefold()):
        map_name = f"map_{country.iso3}.png"
        flag_name = f"flag_{country.iso3}.png"
        map_path = maps_dir / map_name
        flag_path = flags_dir / flag_name
        map_ok = map_path.exists()
        flag_ok = flag_path.exists()
        status = "OK" if map_ok and flag_ok else "MISSING"

        map_src = f"../media/maps/{map_name}"
        flag_src = f"../media/flags/{flag_name}"
        rows.append(
            "\n".join(
                [
                    "<div class='card'>",
                    f"  <h3>{escape(country.name_en)} ({country.iso3})</h3>",
                    f"  <p class='status {status.lower()}'>{status}</p>",
                    f"  <img src='{escape(map_src)}' alt='Map {escape(country.name_en)}' width='{thumbnail_width_px}'>",
                    f"  <img src='{escape(flag_src)}' alt='Flag {escape(country.name_en)}' width='{int(thumbnail_width_px * 0.6)}'>",
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
            f"    .grid {{ display: grid; grid-template-columns: repeat({max_columns}, minmax(220px, 1fr)); gap: 16px; }}",
            "    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }",
            "    .card h3 { margin: 0 0 8px 0; font-size: 16px; }",
            "    .status { margin: 0 0 8px 0; font-weight: 700; }",
            "    .status.ok { color: #197a2f; }",
            "    .status.missing { color: #b22d2d; }",
            "    img { display: block; max-width: 100%; margin-bottom: 8px; }",
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
