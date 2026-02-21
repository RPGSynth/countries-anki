"""Interactive inspection report generation for debugging data flow."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Sequence

from .city_selection import compute_tiny_metrics, determine_target_n
from .cities import CitySelectionError, CitySelector, load_cities_overrides, names_match
from .config import AppConfig
from .countries import load_un_members
from .io_ne import NaturalEarthRepository
from .models import CityOverride, CityRecord, CountrySpec
from .util import write_json


def generate_inspection_report(
    cfg: AppConfig,
    *,
    country_filters: Sequence[str],
    limit: int,
) -> tuple[Path, Path]:
    """Generate a visual + JSON debug report for loaded data and city selection."""
    countries = load_un_members(cfg.paths.un_members)
    country_index = {country.iso3: country for country in countries}
    overrides = load_cities_overrides(cfg.paths.cities_overrides)
    selector = CitySelector(default_n=cfg.cities.default_n)

    filtered_countries = _filter_countries(countries, country_filters=country_filters, limit=limit)
    filter_iso = {country.iso3 for country in filtered_countries}

    repo = NaturalEarthRepository(cfg.paths.ne_admin0_countries, cfg.paths.ne_populated_places)
    admin0_df = repo.load_admin0()
    places_df = repo.load_populated_places()

    iso_allowlist = set(country_index.keys())
    admin_iso_col = repo.detect_admin0_iso_column(admin0_df, iso_allowlist=iso_allowlist)
    places_iso_col = repo.detect_places_iso_column(places_df, iso_allowlist=iso_allowlist)

    rows: list[dict[str, Any]] = []
    for country in filtered_countries:
        row = _analyze_country(
            country=country,
            repo=repo,
            admin0_df=admin0_df,
            places_df=places_df,
            admin_iso_col=admin_iso_col,
            places_iso_col=places_iso_col,
            selector=selector,
            override=overrides.get(country.iso3),
            capital_fields=cfg.cities.capital_fields,
            default_n=cfg.cities.default_n,
            min_n_for_tiny=cfg.cities.min_n_for_tiny,
            tiny_bbox_width=cfg.cities.tiny_country_bbox_deg.width,
            tiny_bbox_height=cfg.cities.tiny_country_bbox_deg.height,
            tiny_area=cfg.cities.tiny_country_area_deg2,
            maps_dir=cfg.paths.maps_dir,
        )
        rows.append(row)

    summary = _build_summary(rows)
    payload = {
        "meta": {
            "countries_total": len(countries),
            "countries_in_report": len(filtered_countries),
            "filters": sorted(filter_iso),
            "admin0_iso_column": admin_iso_col,
            "places_iso_column": places_iso_col,
            "admin0_row_count": int(len(admin0_df)),
            "places_row_count": int(len(places_df)),
        },
        "summary": summary,
        "countries": rows,
    }

    json_path = cfg.paths.manifests_dir / "inspect_report.json"
    html_path = cfg.paths.qa_dir / "inspect.html"
    write_json(json_path, payload)
    _write_html_report(payload=payload, output_html=html_path)
    return (html_path, json_path)


def _filter_countries(
    countries: Sequence[CountrySpec],
    *,
    country_filters: Sequence[str],
    limit: int,
) -> list[CountrySpec]:
    if limit < 1:
        raise ValueError("--limit must be >= 1")
    by_iso = {country.iso3: country for country in countries}
    if country_filters:
        selected: list[CountrySpec] = []
        missing: list[str] = []
        for iso_raw in country_filters:
            iso3 = iso_raw.strip().upper()
            country = by_iso.get(iso3)
            if country is None:
                missing.append(iso3)
                continue
            selected.append(country)
        if missing:
            missing_joined = ", ".join(missing)
            raise ValueError(f"Unknown ISO3 in --country filter: {missing_joined}")
        return selected
    return sorted(countries, key=lambda c: c.iso3)[:limit]


def _analyze_country(
    *,
    country: CountrySpec,
    repo: NaturalEarthRepository,
    admin0_df: Any,
    places_df: Any,
    admin_iso_col: str,
    places_iso_col: str,
    selector: CitySelector,
    override: CityOverride | None,
    capital_fields: Sequence[str],
    default_n: int,
    min_n_for_tiny: int,
    tiny_bbox_width: float,
    tiny_bbox_height: float,
    tiny_area: float,
    maps_dir: Path,
) -> dict[str, Any]:
    geometry_rows = repo.extract_country_geometry(admin0_df, country.iso3, iso_col=admin_iso_col)
    geometry_row_count = int(len(geometry_rows))
    geometry_notnull = int(geometry_rows["geometry"].notna().sum()) if "geometry" in geometry_rows else 0
    tiny_metrics = compute_tiny_metrics(
        geometry_rows,
        bbox_width_threshold=tiny_bbox_width,
        bbox_height_threshold=tiny_bbox_height,
        area_threshold=tiny_area,
    )
    target_n, n_source = determine_target_n(
        default_n=default_n,
        min_n_for_tiny=min_n_for_tiny,
        override=override,
        tiny_metrics=tiny_metrics,
    )

    candidates = repo.extract_city_candidates(
        places_df,
        iso3=country.iso3,
        capital_fields=capital_fields,
        iso_col=places_iso_col,
    )
    expected_capital = country.capital
    expected_capital_found = any(names_match(city.name, expected_capital) for city in candidates)

    selection_error: str | None = None
    selected_capital: CityRecord | None = None
    selected_others: list[CityRecord] = []
    capital_resolution = "unresolved"
    try:
        selected = selector.select_for_country(
            iso3=country.iso3,
            canonical_capital=country.capital,
            candidates=candidates,
            override=override,
            target_n=target_n,
        )
        selected_capital = selected.capital
        selected_others = list(selected.others)
        capital_resolution = selected.capital_resolution
    except CitySelectionError as exc:
        selection_error = str(exc)

    if override is None:
        override_payload: dict[str, Any] | None = None
    else:
        override_payload = {
            "n": override.n,
            "capital": override.capital,
            "manual_capital": (
                {
                    "name": override.manual_capital.name,
                    "lon": override.manual_capital.lon,
                    "lat": override.manual_capital.lat,
                }
                if override.manual_capital is not None
                else None
            ),
            "force_include": list(override.force_include),
            "force_exclude": list(override.force_exclude),
        }

    map_name = f"map_{country.iso3}.png"
    map_path = maps_dir / map_name

    return {
        "iso3": country.iso3,
        "name_en": country.name_en,
        "geometry_row_count": geometry_row_count,
        "geometry_non_null_count": geometry_notnull,
        "candidate_city_count": len(candidates),
        "expected_capital": expected_capital,
        "expected_capital_found_in_dataset": expected_capital_found,
        "n_effective": target_n,
        "n_source": n_source,
        "tiny_country": bool(tiny_metrics.is_tiny) if tiny_metrics is not None else False,
        "capital_resolution": capital_resolution,
        "selected_capital": _city_to_dict(selected_capital),
        "selected_others": [_city_to_dict(city) for city in selected_others],
        "selection_error": selection_error,
        "override": override_payload,
        "candidate_preview": [_city_to_dict(city) for city in sorted(candidates, key=_city_debug_sort)[:12]],
        "map_name": map_name,
        "map_exists": map_path.exists(),
    }


def _city_debug_sort(city: CityRecord) -> tuple[float, float, str]:
    scalerank = float(city.scalerank) if city.scalerank is not None else float("inf")
    pop_desc = -float(city.pop_max) if city.pop_max is not None else float("inf")
    return (scalerank, pop_desc, city.name.casefold())


def _build_summary(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    total = len(rows)
    with_selection_error = sum(1 for row in rows if row["selection_error"] is not None)
    with_multi_geometry = sum(1 for row in rows if int(row["geometry_row_count"]) > 1)
    with_no_geometry = sum(1 for row in rows if int(row["geometry_row_count"]) == 0)
    with_tiny_country = sum(1 for row in rows if bool(row["tiny_country"]))
    with_map = sum(1 for row in rows if bool(row["map_exists"]))
    with_missing_expected_capital = sum(
        1 for row in rows if not bool(row["expected_capital_found_in_dataset"])
    )
    with_manual_override = sum(
        1
        for row in rows
        if row["override"] is not None and row["override"].get("manual_capital") is not None
    )
    return {
        "total": total,
        "with_selection_error": with_selection_error,
        "with_multi_geometry": with_multi_geometry,
        "with_no_geometry": with_no_geometry,
        "with_tiny_country": with_tiny_country,
        "with_map": with_map,
        "with_missing_expected_capital": with_missing_expected_capital,
        "with_manual_override": with_manual_override,
    }


def _write_html_report(*, payload: dict[str, Any], output_html: Path) -> None:
    summary = payload["summary"]
    meta = payload["meta"]
    rows = payload["countries"]

    table_rows: list[str] = []
    for row in rows:
        status = "ok" if row["selection_error"] is None else "error"
        selected_capital = row["selected_capital"]["name"] if row["selected_capital"] else "-"
        expected_found = "yes" if row["expected_capital_found_in_dataset"] else "no"
        map_preview = (
            f"<img src='../media/maps/{escape(row['map_name'])}' alt='Map {escape(row['iso3'])}' width='180'>"
            if row["map_exists"]
            else "<span class='muted'>not rendered</span>"
        )
        override_type = "-"
        if row["override"]:
            if row["override"]["manual_capital"] is not None:
                override_type = "manual_capital"
            elif row["override"]["capital"] is not None:
                override_type = "capital"
            else:
                override_type = "other"

        details_json = json.dumps(
            {
                "n_effective": row["n_effective"],
                "n_source": row["n_source"],
                "tiny_country": row["tiny_country"],
                "capital_resolution": row["capital_resolution"],
                "candidate_preview": row["candidate_preview"],
                "selected_others": row["selected_others"],
                "override": row["override"],
            },
            ensure_ascii=False,
            indent=2,
        )
        details_html = (
            "<details><summary>details</summary>"
            f"<pre>{escape(details_json)}</pre>"
            "</details>"
        )
        table_rows.append(
            "\n".join(
                [
                    "<tr>",
                    f"  <td>{escape(row['iso3'])}</td>",
                    f"  <td>{escape(row['name_en'])}</td>",
                    f"  <td>{row['geometry_row_count']}</td>",
                    f"  <td>{row['candidate_city_count']}</td>",
                    f"  <td>{escape(row['expected_capital'])}</td>",
                    f"  <td>{escape(expected_found)}</td>",
                    f"  <td>{map_preview}</td>",
                    f"  <td>{escape(selected_capital)}</td>",
                    f"  <td>{escape(override_type)}</td>",
                    f"  <td class='{status}'>{escape(row['selection_error'] or 'OK')}</td>",
                    f"  <td>{details_html}</td>",
                    "</tr>",
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
            "  <title>countries-anki inspect report</title>",
            "  <style>",
            "    body { font-family: Segoe UI, Arial, sans-serif; margin: 20px; color: #111; }",
            "    h1, h2 { margin: 0 0 12px 0; }",
            "    .meta, .summary { margin: 0 0 18px 0; padding: 12px; border: 1px solid #ddd; border-radius: 8px; }",
            "    .summary-grid { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 8px; }",
            "    .kpi { background: #f7f7f7; border-radius: 6px; padding: 8px; font-weight: 600; }",
            "    table { border-collapse: collapse; width: 100%; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left; }",
            "    th { background: #f4f4f4; }",
            "    td.ok { color: #1f7a1f; font-weight: 700; }",
            "    td.error { color: #b22d2d; font-weight: 700; }",
            "    .muted { color: #666; font-size: 12px; }",
            "    table img { border: 1px solid #ddd; border-radius: 4px; background: #fff; }",
            "    details > pre { white-space: pre-wrap; margin: 8px 0 0 0; background: #fafafa; padding: 8px; border-radius: 6px; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Deckbuilder Inspect Report</h1>",
            "  <div class='meta'>",
            "    <h2>Metadata</h2>",
            f"    <div>Countries total: {meta['countries_total']}</div>",
            f"    <div>Countries in report: {meta['countries_in_report']}</div>",
            f"    <div>Admin0 ISO column: {escape(str(meta['admin0_iso_column']))}</div>",
            f"    <div>Populated places ISO column: {escape(str(meta['places_iso_column']))}</div>",
            f"    <div>Admin0 rows: {meta['admin0_row_count']}</div>",
            f"    <div>Populated places rows: {meta['places_row_count']}</div>",
            "  </div>",
            "  <div class='summary'>",
            "    <h2>Summary</h2>",
            "    <div class='summary-grid'>",
            f"      <div class='kpi'>Total: {summary['total']}</div>",
            f"      <div class='kpi'>Selection errors: {summary['with_selection_error']}</div>",
            f"      <div class='kpi'>Multi geometry: {summary['with_multi_geometry']}</div>",
            f"      <div class='kpi'>No geometry: {summary['with_no_geometry']}</div>",
            f"      <div class='kpi'>Tiny country (auto-N eligible): {summary['with_tiny_country']}</div>",
            f"      <div class='kpi'>Rendered maps available: {summary['with_map']}</div>",
            f"      <div class='kpi'>Missing expected capital in dataset: {summary['with_missing_expected_capital']}</div>",
            f"      <div class='kpi'>Manual overrides: {summary['with_manual_override']}</div>",
            "    </div>",
            "  </div>",
            "  <table>",
            "    <thead>",
            "      <tr>",
            "        <th>ISO3</th>",
            "        <th>Country</th>",
            "        <th>Geom Rows</th>",
            "        <th>City Candidates</th>",
            "        <th>Expected Capital</th>",
            "        <th>Expected Found</th>",
            "        <th>Map</th>",
            "        <th>Selected Capital</th>",
            "        <th>Override</th>",
            "        <th>Selection Status</th>",
            "        <th>Details</th>",
            "      </tr>",
            "    </thead>",
            "    <tbody>",
            *table_rows,
            "    </tbody>",
            "  </table>",
            "</body>",
            "</html>",
            "",
        ]
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def _city_to_dict(city: CityRecord | None) -> dict[str, Any] | None:
    if city is None:
        return None
    return {
        "name": city.name,
        "iso3": city.iso3,
        "lon": city.lon,
        "lat": city.lat,
        "scalerank": city.scalerank,
        "pop_max": city.pop_max,
        "is_capital": city.is_capital,
    }
