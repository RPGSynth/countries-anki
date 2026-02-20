"""Milestone 2 city-selection pipeline and audit artifact writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .cities import CitySelectionError, CitySelector, load_cities_overrides
from .config import AppConfig
from .countries import load_un_members
from .io_ne import NaturalEarthRepository
from .models import CityOverride, CityRecord
from .util import write_json


@dataclass(frozen=True, slots=True)
class TinyCountryMetrics:
    """Geometry-derived tiny-country metrics in EPSG:4326 units."""

    bbox_width_deg: float
    bbox_height_deg: float
    area_deg2: float
    tiny_by_bbox: bool
    tiny_by_area: bool
    is_tiny: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_width_deg": self.bbox_width_deg,
            "bbox_height_deg": self.bbox_height_deg,
            "area_deg2": self.area_deg2,
            "tiny_by_bbox": self.tiny_by_bbox,
            "tiny_by_area": self.tiny_by_area,
            "is_tiny": self.is_tiny,
        }


@dataclass(slots=True)
class CitySelectionReport:
    """Outcome of the deterministic city-selection pipeline."""

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


def run_city_selection(
    cfg: AppConfig,
    *,
    strict_data_files: bool,
) -> CitySelectionReport:
    """Select deterministic city sets for all UN members and emit an audit JSON."""
    report = CitySelectionReport()
    output_path = cfg.paths.manifests_dir / "city_selection.json"

    try:
        countries = load_un_members(cfg.paths.un_members)
        report.add_info(f"Loaded {len(countries)} UN member records from {cfg.paths.un_members}")
    except Exception as exc:
        report.add_error(f"Failed parsing UN members file '{cfg.paths.un_members}': {exc}")
        return report

    try:
        overrides = load_cities_overrides(cfg.paths.cities_overrides)
        report.add_info(f"Loaded {len(overrides)} city override entries")
    except Exception as exc:
        report.add_error(f"Failed parsing cities overrides: {exc}")
        return report

    if not countries:
        report.add_error("UN members list is empty; city selection cannot proceed.")
        return report

    missing_paths = [
        path
        for path in (cfg.paths.ne_admin0_countries, cfg.paths.ne_populated_places)
        if not path.exists()
    ]
    if missing_paths:
        missing_joined = ", ".join(str(path) for path in missing_paths)
        msg = "Missing Natural Earth dataset file(s): " + missing_joined
        _add_quality_issue(
            report,
            msg,
            strict_data_files=strict_data_files,
            strict_policy=cfg.country_policy.strict,
        )
        if report.errors:
            return report
        report.add_info("Skipping city selection because Natural Earth files are unavailable.")
        report.summary = _empty_summary(len(countries))
        return report

    repo = NaturalEarthRepository(cfg.paths.ne_admin0_countries, cfg.paths.ne_populated_places)
    try:
        admin0_df = repo.load_admin0()
        places_df = repo.load_populated_places()
    except Exception as exc:
        _add_quality_issue(
            report,
            f"Failed loading Natural Earth datasets: {exc}",
            strict_data_files=strict_data_files,
            strict_policy=cfg.country_policy.strict,
        )
        return report

    iso_allowlist = {country.iso3 for country in countries}
    try:
        admin_iso_col = repo.detect_admin0_iso_column(admin0_df, iso_allowlist=iso_allowlist)
        places_iso_col = repo.detect_places_iso_column(places_df, iso_allowlist=iso_allowlist)
    except Exception as exc:
        _add_quality_issue(
            report,
            f"Natural Earth schema validation failed: {exc}",
            strict_data_files=strict_data_files,
            strict_policy=cfg.country_policy.strict,
        )
        return report
    report.add_info(
        "Natural Earth ISO columns selected: "
        f"admin0={admin_iso_col}, populated_places={places_iso_col}"
    )

    selector = CitySelector(default_n=cfg.cities.default_n)
    selection_rows: list[dict[str, Any]] = []

    countries_missing_geometry: list[str] = []
    selection_error_codes: list[str] = []
    tiny_auto_reduced = 0
    override_n_used = 0
    manual_capital_used = 0
    auto_relabel_used = 0

    for country in sorted(countries, key=lambda item: item.iso3):
        override = overrides.get(country.iso3)
        geometry_rows = repo.extract_country_geometry(
            admin0_df,
            country.iso3,
            iso_col=admin_iso_col,
        )
        tiny_metrics = compute_tiny_metrics(
            geometry_rows,
            bbox_width_threshold=cfg.cities.tiny_country_bbox_deg.width,
            bbox_height_threshold=cfg.cities.tiny_country_bbox_deg.height,
            area_threshold=cfg.cities.tiny_country_area_deg2,
        )
        if tiny_metrics is None:
            countries_missing_geometry.append(country.iso3)

        target_n, n_source = determine_target_n(
            default_n=cfg.cities.default_n,
            min_n_for_tiny=cfg.cities.min_n_for_tiny,
            override=override,
            tiny_metrics=tiny_metrics,
        )
        if n_source == "override":
            override_n_used += 1
        elif n_source == "tiny_auto":
            tiny_auto_reduced += 1

        candidates = repo.extract_city_candidates(
            places_df,
            iso3=country.iso3,
            capital_fields=cfg.cities.capital_fields,
            iso_col=places_iso_col,
        )

        selection_error: str | None = None
        selected_capital: CityRecord | None = None
        selected_others: tuple[CityRecord, ...] = ()
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
            selected_others = selected.others
            capital_resolution = selected.capital_resolution
            if selected.capital_resolution == "manual_override":
                manual_capital_used += 1
            elif selected.capital_resolution == "auto_relabel":
                auto_relabel_used += 1
        except CitySelectionError as exc:
            selection_error = str(exc)
            selection_error_codes.append(country.iso3)

        selection_rows.append(
            {
                "iso3": country.iso3,
                "name_en": country.name_en,
                "configured_capital": country.capital,
                "candidate_city_count": len(candidates),
                "n_effective": target_n,
                "n_source": n_source,
                "tiny_country": bool(tiny_metrics.is_tiny) if tiny_metrics is not None else False,
                "tiny_metrics": tiny_metrics.to_dict() if tiny_metrics is not None else None,
                "capital_resolution": capital_resolution,
                "selected_capital": _city_to_dict(selected_capital),
                "selected_others": [_city_to_dict(city) for city in selected_others],
                "selection_error": selection_error,
                "override": _override_to_dict(override),
            }
        )

    summary = {
        "countries_total": len(countries),
        "countries_with_selection_errors": len(selection_error_codes),
        "tiny_countries_auto_reduced": tiny_auto_reduced,
        "override_n_used": override_n_used,
        "manual_capital_used": manual_capital_used,
        "auto_relabel_used": auto_relabel_used,
    }
    report.summary = summary

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "countries_total": len(countries),
            "admin0_iso_column": admin_iso_col,
            "places_iso_column": places_iso_col,
            "default_n": cfg.cities.default_n,
            "min_n_for_tiny": cfg.cities.min_n_for_tiny,
            "tiny_thresholds": {
                "bbox_width_deg": cfg.cities.tiny_country_bbox_deg.width,
                "bbox_height_deg": cfg.cities.tiny_country_bbox_deg.height,
                "area_deg2": cfg.cities.tiny_country_area_deg2,
            },
        },
        "summary": summary,
        "countries": selection_rows,
    }
    write_json(output_path, payload)
    report.output_path = output_path
    report.add_info(f"City selection audit written to {output_path}")

    if selection_error_codes:
        report.add_error(
            "City selection failed for: " + _format_code_list(sorted(selection_error_codes))
        )
    if countries_missing_geometry:
        report.add_warning(
            "No geometry available for tiny-country metrics; default N was used for: "
            + _format_code_list(sorted(set(countries_missing_geometry)))
        )
    return report


def determine_target_n(
    *,
    default_n: int,
    min_n_for_tiny: int,
    override: CityOverride | None,
    tiny_metrics: TinyCountryMetrics | None,
) -> tuple[int, str]:
    if override is not None and override.n is not None:
        return (override.n, "override")
    if tiny_metrics is not None and tiny_metrics.is_tiny:
        return (min(default_n, min_n_for_tiny), "tiny_auto")
    return (default_n, "default")


def compute_tiny_metrics(
    geometry_rows: Any,
    *,
    bbox_width_threshold: float,
    bbox_height_threshold: float,
    area_threshold: float,
) -> TinyCountryMetrics | None:
    if "geometry" not in geometry_rows:
        return None
    geometries = geometry_rows["geometry"].dropna()
    if len(geometries) == 0:
        return None

    bounds = geometries.total_bounds
    min_x = float(bounds[0])
    min_y = float(bounds[1])
    max_x = float(bounds[2])
    max_y = float(bounds[3])
    width = max(0.0, max_x - min_x)
    height = max(0.0, max_y - min_y)
    area_deg2 = _sum_geometry_area_deg2(geometries)
    tiny_by_bbox = width < bbox_width_threshold or height < bbox_height_threshold
    tiny_by_area = area_deg2 < area_threshold
    return TinyCountryMetrics(
        bbox_width_deg=width,
        bbox_height_deg=height,
        area_deg2=area_deg2,
        tiny_by_bbox=tiny_by_bbox,
        tiny_by_area=tiny_by_area,
        is_tiny=tiny_by_bbox or tiny_by_area,
    )


def _sum_geometry_area_deg2(geometries: Any) -> float:
    """Approximate area in degree^2 without GeoSeries.area CRS warning."""
    area_sum = 0.0
    for geometry in geometries:
        if geometry is None:
            continue
        if hasattr(geometry, "is_empty") and bool(geometry.is_empty):
            continue
        area_sum += float(geometry.area)
    return area_sum


def _override_to_dict(override: CityOverride | None) -> dict[str, Any] | None:
    if override is None:
        return None
    return {
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
        "label_overrides": dict(override.label_overrides),
    }


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


def _empty_summary(country_count: int) -> dict[str, int]:
    return {
        "countries_total": country_count,
        "countries_with_selection_errors": 0,
        "tiny_countries_auto_reduced": 0,
        "override_n_used": 0,
        "manual_capital_used": 0,
        "auto_relabel_used": 0,
    }


def _add_quality_issue(
    report: CitySelectionReport,
    msg: str,
    *,
    strict_data_files: bool,
    strict_policy: bool,
) -> None:
    if strict_data_files or strict_policy:
        report.add_error(msg)
    else:
        report.add_warning(msg)


def _format_code_list(values: Sequence[str], limit: int = 12) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    shown = ", ".join(values[:limit])
    return f"{shown}, ... (+{len(values) - limit} more)"


def format_city_selection_lines(report: CitySelectionReport) -> Sequence[str]:
    lines: list[str] = []
    lines.extend(f"[INFO] {msg}" for msg in report.infos)
    lines.extend(f"[WARN] {msg}" for msg in report.warnings)
    lines.extend(f"[ERROR] {msg}" for msg in report.errors)
    if report.summary:
        summary = report.summary
        lines.append(
            "[INFO] City selection summary: "
            f"countries_total={summary['countries_total']}, "
            f"selection_errors={summary['countries_with_selection_errors']}, "
            f"tiny_auto_reduced={summary['tiny_countries_auto_reduced']}, "
            f"override_n_used={summary['override_n_used']}, "
            f"manual_capital_used={summary['manual_capital_used']}, "
            f"auto_relabel_used={summary['auto_relabel_used']}"
        )
    if report.ok:
        lines.append("[OK] City selection completed with no errors.")
    return lines
