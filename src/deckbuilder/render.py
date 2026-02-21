"""Milestone 3 map rendering pipeline for country maps."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .city_selection import run_city_selection
from .config import AppConfig, RenderConfig
from .countries import load_un_members
from .io_ne import NaturalEarthRepository
from .models import CityRecord, RenderOverride, SelectedCities
from .render_overrides import load_render_overrides


_BACKGROUND_WHITE = "white"
_BACKGROUND_SATELLITE = "satellite"

_LOGGER = logging.getLogger("deckbuilder.render")


@dataclass(frozen=True, slots=True)
class _GeometryDrawPolicy:
    context_line_alpha: float
    context_line_style: tuple[Any, ...]
    context_line_color: str
    segment_jump_threshold_m: float


@dataclass(frozen=True, slots=True)
class _InsetPolicy:
    max_count: int
    distance_deg: float
    very_far_deg: float
    min_area_share: float
    city_match_eps: float
    box_size: float
    box_gap: float
    box_margin: float


@dataclass(frozen=True, slots=True)
class _ExtentPolicy:
    deg_to_m: float
    tiny_country_max_m: float
    tiny_padding_x_scale: float
    tiny_padding_y_scale: float
    tiny_padding_x_bias_main: float
    tiny_padding_y_bias_main: float
    tiny_padding_x_bias_inset: float
    tiny_padding_y_bias_inset: float
    main_min_padding_floor_m: float
    inset_min_padding_floor_m: float
    main_min_span_m: float
    inset_min_span_m: float
    inset_padding_scale: float
    capital_center_half_height_m: float


@dataclass(frozen=True, slots=True)
class _BasemapPolicy:
    main_connections: int
    inset_connections: int


@dataclass(frozen=True, slots=True)
class _LabelPolicy:
    city_visual_scale: float
    edge_guard_ratio: float
    edge_guard_min_main_m: float
    edge_guard_min_inset_m: float


# Rendering heuristics are grouped by concern so tuning stays localized.
_GEOMETRY_DRAW_POLICY = _GeometryDrawPolicy(
    context_line_alpha=0.46,
    context_line_style=(0, (1.8, 2.8)),
    context_line_color="#777777",
    segment_jump_threshold_m=2_500_000.0,
)
_INSET_POLICY = _InsetPolicy(
    max_count=3,
    distance_deg=9.0,
    very_far_deg=24.0,
    min_area_share=0.003,
    city_match_eps=0.03,
    box_size=0.20,
    box_gap=0.015,
    box_margin=0.02,
)
_EXTENT_POLICY = _ExtentPolicy(
    deg_to_m=111_320.0,
    tiny_country_max_m=1.5 * 111_320.0,
    tiny_padding_x_scale=0.16,
    tiny_padding_y_scale=0.16,
    tiny_padding_x_bias_main=900.0,
    tiny_padding_y_bias_main=900.0,
    tiny_padding_x_bias_inset=600.0,
    tiny_padding_y_bias_inset=600.0,
    main_min_padding_floor_m=900.0,
    inset_min_padding_floor_m=550.0,
    main_min_span_m=6_000.0,
    inset_min_span_m=3_500.0,
    inset_padding_scale=0.9,
    capital_center_half_height_m=140_000.0,
)
_BASEMAP_POLICY = _BasemapPolicy(main_connections=4, inset_connections=2)
_LABEL_POLICY = _LabelPolicy(
    city_visual_scale=1.25,
    edge_guard_ratio=0.045,
    edge_guard_min_main_m=1_500.0,
    edge_guard_min_inset_m=800.0,
)


@dataclass(frozen=True, slots=True)
class RenderRequest:
    iso3: str
    main_geometry: Any
    inset_geometries: tuple[Any, ...]
    context_geometries: tuple[Any, ...]
    selected_cities: SelectedCities
    output_path: Path
    label_overrides: Mapping[str, str] = field(default_factory=dict)
    render_override: RenderOverride | None = None


@dataclass(frozen=True, slots=True)
class _SelectionEntry:
    selected_cities: SelectedCities
    label_overrides: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _CountryRenderPolicy:
    draw_outline: bool
    center_on_capital: bool
    use_insets: bool
    capital_zoom_pct: float | None


@dataclass(frozen=True, slots=True)
class _LocalProjection:
    """Global Web Mercator projection (EPSG:3857)."""

    transformer: Any


@dataclass(frozen=True, slots=True)
class _ProjectedCity:
    name: str
    source_name: str
    x: float
    y: float
    is_capital: bool


@dataclass(frozen=True, slots=True)
class _PreparedRenderInputs:
    main_projection: _LocalProjection
    main_geometry: Any
    context_geometries: tuple[Any, ...]
    inset_geometries: tuple[Any, ...]
    main_projected_cities: tuple[_ProjectedCity, ...]
    inset_city_items: tuple[tuple[tuple[CityRecord, bool], ...], ...]


@dataclass(frozen=True, slots=True)
class _PanelRenderSpec:
    ax: Any
    extent: tuple[float, float, float, float]
    geometry: Any
    cities: tuple[_ProjectedCity, ...]
    label_overrides: Mapping[str, str]
    draw_outline: bool
    is_inset: bool
    context_geometries: tuple[Any, ...] = ()


_PixelBBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _LabelCandidate:
    city: _ProjectedCity
    text: str


@dataclass(frozen=True, slots=True)
class _ExtentModeSettings:
    min_padding_floor_m: float
    min_span_m: float
    pad_scale: float
    target_ratio: float


@dataclass(slots=True)
class RenderMapsReport:
    output_dir: Path | None = None
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


class MapRenderer:
    """Deterministic renderer for one country map PNG."""

    def __init__(self, cfg: RenderConfig, *, debug_render: bool = False) -> None:
        self.cfg = cfg
        self.debug_render = debug_render
        self._basemap_source = _resolve_basemap_source(cfg.background.mode)
        self._basemap_failure: str | None = None

    @property
    def basemap_warning(self) -> str | None:
        return self._basemap_failure

    def render(self, req: RenderRequest) -> Path:
        plt, transforms = _require_matplotlib()
        width_px = self.cfg.image.width_px
        height_px = self.cfg.image.height_px
        dpi = self.cfg.image.dpi
        policy = _resolve_country_render_policy(
            override=req.render_override,
            background_mode=self.cfg.background.mode,
        )
        prepared = self._prepare_render_inputs(req=req, policy=policy)

        fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        try:
            main_extent = self._resolve_main_extent(
                req=req,
                policy=policy,
                main_projection=prepared.main_projection,
                main_geometry=prepared.main_geometry,
            )
            self._render_panel(
                fig=fig,
                transforms=transforms,
                spec=_PanelRenderSpec(
                    ax=ax,
                    extent=main_extent,
                    geometry=prepared.main_geometry,
                    cities=prepared.main_projected_cities,
                    label_overrides=req.label_overrides,
                    draw_outline=policy.draw_outline,
                    is_inset=False,
                    context_geometries=prepared.context_geometries,
                ),
                apply_figure_background=True,
            )

            self._draw_insets(
                fig=fig,
                transforms=transforms,
                inset_geometries=prepared.inset_geometries,
                inset_city_items=prepared.inset_city_items,
                label_overrides=req.label_overrides,
                draw_outline=policy.draw_outline,
            )

            req.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                req.output_path,
                dpi=dpi,
                format=self.cfg.image.format,
                transparent=self.cfg.image.background.casefold() == "transparent",
            )
            return req.output_path
        finally:
            plt.close(fig)

    def _prepare_render_inputs(
        self,
        *,
        req: RenderRequest,
        policy: _CountryRenderPolicy,
    ) -> _PreparedRenderInputs:
        main_projection = _build_local_projection(req.main_geometry)
        main_geometry = _project_geometry(req.main_geometry, main_projection)
        context_geometries = tuple(
            _project_geometry(geometry, main_projection)
            for geometry in req.context_geometries
        )

        effective_inset_geometries = req.inset_geometries
        if not policy.use_insets:
            effective_inset_geometries = ()
        elif effective_inset_geometries:
            main_geometry, effective_inset_geometries = _merge_visible_insets_into_main_view(
                main_geometry=main_geometry,
                inset_geometries=effective_inset_geometries,
                projection=main_projection,
                cfg=self.cfg,
            )

        main_city_items, inset_city_items = _partition_selected_cities(
            selected=req.selected_cities,
            inset_geometries=effective_inset_geometries,
        )
        return _PreparedRenderInputs(
            main_projection=main_projection,
            main_geometry=main_geometry,
            context_geometries=context_geometries,
            inset_geometries=tuple(effective_inset_geometries),
            main_projected_cities=_project_city_items(main_city_items, main_projection),
            inset_city_items=inset_city_items,
        )

    def _resolve_main_extent(
        self,
        *,
        req: RenderRequest,
        policy: _CountryRenderPolicy,
        main_projection: _LocalProjection,
        main_geometry: Any,
    ) -> tuple[float, float, float, float]:
        if policy.center_on_capital:
            return _compute_capital_centered_extent(
                capital=req.selected_cities.capital,
                cfg=self.cfg,
                projection=main_projection,
                zoom_pct=policy.capital_zoom_pct,
            )
        return _compute_extent(
            geometry=main_geometry,
            cfg=self.cfg,
            projection=main_projection,
            is_inset=False,
        )

    def _render_panel(
        self,
        *,
        fig: Any,
        transforms: Any,
        spec: _PanelRenderSpec,
        apply_figure_background: bool,
    ) -> None:
        _apply_background(
            fig=fig if apply_figure_background else None,
            ax=spec.ax,
            background=self.cfg.image.background,
        )
        self._configure_panel_axes(ax=spec.ax, extent=spec.extent, is_inset=spec.is_inset)
        self._draw_basemap(ax=spec.ax, is_inset=spec.is_inset)
        if spec.is_inset:
            self._style_inset_frame(spec.ax)
        if spec.context_geometries:
            _draw_context_outlines(ax=spec.ax, geometries=spec.context_geometries)
        if spec.draw_outline:
            _draw_geometry_outline(
                ax=spec.ax,
                geometry=spec.geometry,
                color=self.cfg.style.country_outline_color,
                line_width=self._outline_line_width(spec.is_inset),
                zorder=2,
            )
        self._draw_panel_city_content(
            ax=spec.ax,
            fig=fig,
            transforms=transforms,
            cities=spec.cities,
            label_overrides=spec.label_overrides,
        )

    def _draw_panel_city_content(
        self,
        *,
        ax: Any,
        fig: Any,
        transforms: Any,
        cities: tuple[_ProjectedCity, ...],
        label_overrides: Mapping[str, str],
    ) -> None:
        _draw_city_markers(
            ax=ax,
            cities=cities,
            marker_color=self.cfg.style.city_marker_color,
            city_marker_size=self.cfg.style.city_marker_size,
            capital_marker_size=self.cfg.style.capital_marker_size,
            zorder=4,
        )
        _place_city_labels(
            ax=ax,
            fig=fig,
            transforms=transforms,
            cities=cities,
            label_overrides=label_overrides,
            cfg=self.cfg,
            zorder_base=5,
        )

    def _configure_panel_axes(
        self,
        *,
        ax: Any,
        extent: tuple[float, float, float, float],
        is_inset: bool,
    ) -> None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        if is_inset:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    def _style_inset_frame(self, ax: Any) -> None:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_edgecolor(self.cfg.style.country_outline_color)

    def _outline_line_width(self, is_inset: bool) -> float:
        if is_inset:
            return max(self.cfg.style.country_outline_width * 0.9, 0.9)
        return self.cfg.style.country_outline_width

    def _draw_insets(
        self,
        *,
        fig: Any,
        transforms: Any,
        inset_geometries: Sequence[Any],
        inset_city_items: Sequence[tuple[tuple[CityRecord, bool], ...]],
        label_overrides: Mapping[str, str],
        draw_outline: bool,
    ) -> None:
        if not inset_geometries:
            return

        positions = _inset_positions(len(inset_geometries))
        for idx, geometry in enumerate(inset_geometries):
            if idx >= len(positions) or not _is_valid_geometry(geometry):
                continue

            inset_projection = _build_local_projection(geometry)
            projected_geometry = _project_geometry(geometry, inset_projection)
            city_items = inset_city_items[idx] if idx < len(inset_city_items) else ()
            cities_projected = _project_city_items(city_items, inset_projection)

            ax_inset = fig.add_axes(list(positions[idx]), zorder=20 + idx)
            extent = _compute_extent(
                geometry=projected_geometry,
                cfg=self.cfg,
                projection=inset_projection,
                is_inset=True,
            )
            self._render_panel(
                fig=fig,
                transforms=transforms,
                spec=_PanelRenderSpec(
                    ax=ax_inset,
                    extent=extent,
                    geometry=projected_geometry,
                    cities=cities_projected,
                    label_overrides=label_overrides,
                    draw_outline=draw_outline,
                    is_inset=True,
                ),
                apply_figure_background=False,
            )

    def _draw_basemap(self, *, ax: Any, is_inset: bool) -> None:
        if self._basemap_source is None or self._basemap_failure is not None:
            return
        contextily = _require_contextily()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        n_connections = (
            _BASEMAP_POLICY.inset_connections
            if is_inset
            else _BASEMAP_POLICY.main_connections
        )
        try:
            if self.debug_render:
                _LOGGER.info(
                    "[render-debug] basemap fetch (%s): x=[%.1f, %.1f], y=[%.1f, %.1f]",
                    "inset" if is_inset else "main",
                    x0,
                    x1,
                    y0,
                    y1,
                )
            image, extent = contextily.bounds2img(
                x0,
                y0,
                x1,
                y1,
                zoom="auto",
                source=self._basemap_source,
                ll=False,
                use_cache=True,
                n_connections=n_connections,
                max_retries=1,
            )
            ax.imshow(
                image,
                extent=extent,
                interpolation="bilinear",
                zorder=-8,
                alpha=1.0,
            )
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            if self.debug_render:
                _LOGGER.info("[render-debug] basemap fetch complete (%s)", "inset" if is_inset else "main")
        except Exception as exc:
            self._basemap_failure = (
                "Basemap loading failed once and was disabled for remaining renders: "
                f"{exc}"
            )
            _LOGGER.warning(self._basemap_failure)


def run_render_maps(
    cfg: AppConfig,
    *,
    strict_data_files: bool,
    selection_manifest_path: Path | None = None,
    country_filter: Sequence[str] | None = None,
    limit_countries: int | None = None,
    debug_render: bool = False,
) -> RenderMapsReport:
    """Render maps for all UN members from selected cities + Natural Earth geometry."""
    report = RenderMapsReport(output_dir=cfg.paths.maps_dir)
    countries = load_un_members(cfg.paths.un_members)
    report.add_info(f"Loaded {len(countries)} UN member records from {cfg.paths.un_members}")
    if not countries:
        report.add_error("UN members list is empty; cannot render maps.")
        return report
    if limit_countries is not None and limit_countries < 1:
        report.add_error("limit_countries must be >= 1 when provided.")
        return report

    countries = sorted(countries, key=lambda item: item.iso3)
    if country_filter:
        requested = {item.strip().upper() for item in country_filter if item and item.strip()}
        if requested:
            countries = [country for country in countries if country.iso3 in requested]
            report.add_info(
                f"Country filter enabled: {len(countries)} selected from {len(requested)} requested codes."
            )
            missing_requested = sorted(requested - {country.iso3 for country in countries})
            if missing_requested:
                report.add_warning(
                    "Requested ISO3 not present in UN list: "
                    + _format_code_list(missing_requested)
                )
    if limit_countries is not None:
        countries = countries[:limit_countries]
        report.add_info(f"Country render limit enabled: first {len(countries)} countries.")
    if not countries:
        report.add_error("No countries selected for rendering after filters/limits.")
        return report

    if selection_manifest_path is None:
        city_report = run_city_selection(cfg, strict_data_files=strict_data_files)
        report.infos.extend(f"[city-selection] {line}" for line in city_report.infos)
        report.warnings.extend(f"[city-selection] {line}" for line in city_report.warnings)
        if not city_report.ok:
            report.errors.extend(f"[city-selection] {line}" for line in city_report.errors)
            return report
        if city_report.output_path is None:
            report.add_error(
                "City selection did not produce a manifest; rendering cannot continue."
            )
            return report
        selection_manifest_path = city_report.output_path

    if not selection_manifest_path.exists():
        report.add_error(f"City selection manifest not found: {selection_manifest_path}")
        return report
    report.add_info(f"Using city selection manifest: {selection_manifest_path}")

    try:
        selections = _load_selection_manifest(selection_manifest_path)
    except Exception as exc:
        report.add_error(
            f"Failed loading city selection manifest '{selection_manifest_path}': {exc}"
        )
        return report

    try:
        render_overrides = load_render_overrides(cfg.paths.render_overrides)
    except Exception as exc:
        report.add_error(f"Failed loading render overrides '{cfg.paths.render_overrides}': {exc}")
        return report
    report.add_info(f"Loaded {len(render_overrides)} render override entries")

    repo = NaturalEarthRepository(cfg.paths.ne_admin0_countries, cfg.paths.ne_populated_places)
    try:
        admin0_df = repo.load_admin0()
    except Exception as exc:
        report.add_error(f"Failed loading Natural Earth admin0 dataset: {exc}")
        return report

    iso_allowlist = {country.iso3 for country in countries}
    try:
        admin_iso_col = repo.detect_admin0_iso_column(admin0_df, iso_allowlist=iso_allowlist)
    except Exception as exc:
        report.add_error(f"Natural Earth schema validation failed for render: {exc}")
        return report
    report.add_info(f"Natural Earth ISO column selected for rendering: admin0={admin_iso_col}")

    try:
        renderer = MapRenderer(cfg.render, debug_render=debug_render)
    except Exception as exc:
        report.add_error(f"Failed initializing map renderer: {exc}")
        return report
    report.add_info(f"Basemap mode: {cfg.render.background.mode}")
    render_failures: list[str] = []
    rendered_count = 0
    missing_selection: list[str] = []
    missing_geometry: list[str] = []

    for idx, country in enumerate(countries, start=1):
        country_t0 = time.perf_counter()
        if debug_render:
            _LOGGER.info("[render-debug] (%d/%d) %s start", idx, len(countries), country.iso3)
        selection_entry = selections.get(country.iso3)
        if selection_entry is None:
            missing_selection.append(country.iso3)
            if debug_render:
                _LOGGER.info("[render-debug] %s skipped (no city selection entry)", country.iso3)
            continue

        geometry_rows = repo.extract_country_geometry(
            admin0_df,
            country.iso3,
            iso_col=admin_iso_col,
        )
        geometry = _merge_country_geometry(geometry_rows)
        if geometry is None:
            missing_geometry.append(country.iso3)
            if debug_render:
                _LOGGER.info("[render-debug] %s skipped (missing geometry)", country.iso3)
            continue

        render_override = render_overrides.get(country.iso3)
        main_geometry, inset_geometries = _partition_country_geometry_for_insets(
            geometry=geometry,
            selected=selection_entry.selected_cities,
            prefer_capital_polygon_as_main=(
                render_override is not None and render_override.capital_polygon_as_main is True
            ),
        )
        context_geometries = _collect_context_geometries(
            admin0_df=admin0_df,
            target_iso3=country.iso3,
            iso_col=admin_iso_col,
            target_geometry=main_geometry,
        )

        output_path = cfg.paths.maps_dir / f"map_{country.iso3}.png"
        req = RenderRequest(
            iso3=country.iso3,
            main_geometry=main_geometry,
            inset_geometries=inset_geometries,
            context_geometries=context_geometries,
            selected_cities=selection_entry.selected_cities,
            output_path=output_path,
            label_overrides=selection_entry.label_overrides,
            render_override=render_override,
        )
        try:
            renderer.render(req)
            rendered_count += 1
            elapsed = time.perf_counter() - country_t0
            _LOGGER.info(
                "[render] (%d/%d) built %s in %.2fs",
                idx,
                len(countries),
                country.iso3,
                elapsed,
            )
            if debug_render:
                _LOGGER.info(
                    "[render-debug] %s done in %.2fs (insets=%d, override=%s)",
                    country.iso3,
                    elapsed,
                    len(inset_geometries),
                    "yes" if render_override is not None else "no",
                )
        except Exception as exc:
            render_failures.append(f"{country.iso3}({exc})")
            if debug_render:
                elapsed = time.perf_counter() - country_t0
                _LOGGER.exception(
                    "[render-debug] %s failed after %.2fs",
                    country.iso3,
                    elapsed,
                )

    total = len(countries)
    report.summary = {
        "countries_total": total,
        "maps_rendered": rendered_count,
        "maps_failed": len(render_failures),
        "maps_missing_selection": len(missing_selection),
        "maps_missing_geometry": len(missing_geometry),
    }

    if missing_selection:
        report.add_error(
            "Countries missing city selection data: "
            + _format_code_list(sorted(missing_selection))
        )
    if missing_geometry:
        report.add_error(
            "Countries missing valid geometry for rendering: "
            + _format_code_list(sorted(missing_geometry))
        )
    if render_failures:
        report.add_error("Render failures: " + _format_code_list(sorted(render_failures)))
    if renderer.basemap_warning is not None:
        report.add_warning(renderer.basemap_warning)

    report.add_info(
        "Render summary: "
        f"countries_total={total}, "
        f"maps_rendered={rendered_count}, "
        f"maps_failed={len(render_failures)}, "
        f"missing_selection={len(missing_selection)}, "
        f"missing_geometry={len(missing_geometry)}"
    )
    if report.ok:
        report.add_info(f"Rendered map files written to {cfg.paths.maps_dir}")
    return report


def format_render_lines(report: RenderMapsReport) -> Sequence[str]:
    lines: list[str] = []
    lines.extend(f"[INFO] {msg}" for msg in report.infos)
    lines.extend(f"[WARN] {msg}" for msg in report.warnings)
    lines.extend(f"[ERROR] {msg}" for msg in report.errors)
    if report.ok:
        lines.append("[OK] Map rendering completed with no errors.")
    return lines


def _resolve_country_render_policy(
    *,
    override: RenderOverride | None,
    background_mode: str,
) -> _CountryRenderPolicy:
    default_use_insets = True
    default_center_on_capital = False
    default_draw_outline = True

    use_insets = (
        override.inset
        if override is not None and override.inset is not None
        else default_use_insets
    )
    has_zoom_override = override is not None and override.capital_zoom_pct is not None
    center_on_capital = (
        override.center_on_capital
        if override is not None and override.center_on_capital is not None
        else (has_zoom_override or default_center_on_capital)
    )
    draw_outline = (
        override.outline
        if override is not None and override.outline is not None
        else default_draw_outline
    )
    capital_zoom_pct = (
        override.capital_zoom_pct
        if override is not None and override.capital_zoom_pct is not None
        else None
    )
    if background_mode.casefold() == _BACKGROUND_WHITE:
        draw_outline = True
    return _CountryRenderPolicy(
        draw_outline=draw_outline,
        center_on_capital=center_on_capital,
        use_insets=use_insets,
        capital_zoom_pct=capital_zoom_pct,
    )


def _load_selection_manifest(path: Path) -> dict[str, _SelectionEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object at root.")
    rows = payload.get("countries")
    if not isinstance(rows, list):
        raise ValueError("Expected 'countries' list in city selection manifest.")

    out: dict[str, _SelectionEntry] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        iso3_raw = row.get("iso3")
        if not isinstance(iso3_raw, str) or len(iso3_raw.strip()) != 3:
            continue
        iso3 = iso3_raw.strip().upper()
        if row.get("selection_error") is not None:
            continue

        capital_raw = row.get("selected_capital")
        others_raw = row.get("selected_others", [])
        if not isinstance(capital_raw, dict) or not isinstance(others_raw, list):
            continue
        capital = _city_from_dict(capital_raw)
        others = tuple(
            city for city in (_city_from_dict(item) for item in others_raw) if city is not None
        )
        if capital is None:
            continue

        override_raw = row.get("override")
        label_overrides: dict[str, str] = {}
        if isinstance(override_raw, dict):
            labels_raw = override_raw.get("label_overrides")
            if isinstance(labels_raw, dict):
                for key, value in labels_raw.items():
                    if isinstance(key, str) and isinstance(value, str):
                        label_overrides[key] = value

        selected = SelectedCities(
            capital=capital,
            others=others,
            effective_n=row.get("n_effective") if isinstance(row.get("n_effective"), int) else None,
            capital_resolution=(
                row.get("capital_resolution")
                if isinstance(row.get("capital_resolution"), str)
                else "unknown"
            ),
        )
        out[iso3] = _SelectionEntry(selected_cities=selected, label_overrides=label_overrides)
    return out


def _city_from_dict(raw: Any) -> CityRecord | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    iso3 = raw.get("iso3")
    lon = raw.get("lon")
    lat = raw.get("lat")
    if not isinstance(name, str) or not isinstance(iso3, str):
        return None
    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
        return None
    scalerank_raw = raw.get("scalerank")
    pop_max_raw = raw.get("pop_max")
    scalerank = int(scalerank_raw) if isinstance(scalerank_raw, (int, float)) else None
    pop_max = int(pop_max_raw) if isinstance(pop_max_raw, (int, float)) else None
    is_capital = bool(raw.get("is_capital"))
    return CityRecord(
        name=name,
        iso3=iso3.strip().upper(),
        lon=float(lon),
        lat=float(lat),
        scalerank=scalerank,
        pop_max=pop_max,
        is_capital=is_capital,
    )


def _merge_country_geometry(geometry_rows: Any) -> Any | None:
    if "geometry" not in geometry_rows:
        return None
    geometries = [
        geometry
        for geometry in geometry_rows["geometry"]
        if _is_valid_geometry(geometry)
    ]
    if not geometries:
        return None

    unary_union = _require_shapely_unary_union()
    merged = unary_union(geometries)
    if merged is None or not _is_valid_geometry(merged):
        return None
    return merged

def _partition_country_geometry_for_insets(
    *,
    geometry: Any,
    selected: SelectedCities,
    prefer_capital_polygon_as_main: bool = False,
) -> tuple[Any, tuple[Any, ...]]:
    components = _explode_polygons(geometry)
    if len(components) <= 1:
        return (geometry, ())

    ordered_components = sorted(
        components,
        key=lambda item: (
            -float(item.area),
            float(item.centroid.x),
            float(item.centroid.y),
        ),
    )
    main_component = ordered_components[0]
    if prefer_capital_polygon_as_main:
        capital_point = _require_shapely_point_factory()(
            selected.capital.lon,
            selected.capital.lat,
        )
        matched = [
            component
            for component in ordered_components
            if _geometry_matches_point(component, capital_point)
        ]
        if matched:
            main_component = matched[0]

    main_area = max(float(main_component.area), 1e-9)
    city_points = _selected_city_points(selected)

    main_parts: list[Any] = [main_component]
    inset_candidates: list[Any] = []
    for component in components:
        if component is main_component:
            continue
        distance = float(component.distance(main_component))
        area_share = float(component.area) / main_area
        has_selected_city = _geometry_has_selected_city(component, city_points)
        far = distance >= _INSET_POLICY.distance_deg
        very_far = distance >= _INSET_POLICY.very_far_deg
        if far and (
            area_share >= _INSET_POLICY.min_area_share
            or has_selected_city
            or very_far
        ):
            inset_candidates.append(component)
        else:
            main_parts.append(component)

    if not inset_candidates:
        return (geometry, ())

    inset_candidates.sort(
        key=lambda item: (
            -float(item.area),
            float(item.centroid.x),
            float(item.centroid.y),
        )
    )

    if len(inset_candidates) > _INSET_POLICY.max_count:
        keep_count = max(_INSET_POLICY.max_count - 1, 0)
        keep = inset_candidates[:keep_count]
        overflow = inset_candidates[keep_count:]
        if overflow:
            keep.append(_require_shapely_unary_union()(overflow))
        inset_geometries = keep
    else:
        inset_geometries = inset_candidates

    main_geometry = _require_shapely_unary_union()(main_parts)
    if not _is_valid_geometry(main_geometry):
        return (geometry, tuple(inset_geometries))
    return (main_geometry, tuple(inset_geometries))


def _merge_visible_insets_into_main_view(
    *,
    main_geometry: Any,
    inset_geometries: Sequence[Any],
    projection: _LocalProjection,
    cfg: RenderConfig,
) -> tuple[Any, tuple[Any, ...]]:
    if not inset_geometries:
        return (main_geometry, ())
    extent = _compute_extent(
        geometry=main_geometry,
        cfg=cfg,
        projection=projection,
        is_inset=False,
    )
    viewport = _require_shapely_box_factory()(extent[0], extent[2], extent[1], extent[3])

    merge_parts: list[Any] = []
    kept_insets: list[Any] = []
    for inset_geometry in inset_geometries:
        projected_inset = _project_geometry(inset_geometry, projection)
        if not _is_valid_geometry(projected_inset):
            continue
        if projected_inset.intersects(viewport):
            merge_parts.append(projected_inset)
        else:
            kept_insets.append(inset_geometry)

    if merge_parts:
        merged = _require_shapely_unary_union()((main_geometry, *merge_parts))
        if _is_valid_geometry(merged):
            main_geometry = merged
    return (main_geometry, tuple(kept_insets))


def _selected_city_points(selected: SelectedCities) -> tuple[Any, ...]:
    return tuple(_city_to_point(city) for city in (selected.capital, *selected.others))


def _geometry_has_selected_city(geometry: Any, city_points: Sequence[Any]) -> bool:
    if not city_points:
        return False
    for point in city_points:
        if _geometry_matches_point(geometry, point):
            return True
    return False


def _geometry_matches_point(geometry: Any, point: Any) -> bool:
    envelope = geometry.buffer(_INSET_POLICY.city_match_eps)
    return bool(
        envelope.contains(point) or envelope.distance(point) <= _INSET_POLICY.city_match_eps
    )


def _explode_polygons(geometry: Any) -> list[Any]:
    if not _is_valid_geometry(geometry):
        return []
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return [part for part in geometry.geoms if _is_valid_geometry(part)]
    if geom_type == "GeometryCollection":
        out: list[Any] = []
        for part in geometry.geoms:
            out.extend(_explode_polygons(part))
        return out
    return []


def _collect_context_geometries(
    *,
    admin0_df: Any,
    target_iso3: str,
    iso_col: str,
    target_geometry: Any,
) -> tuple[Any, ...]:
    min_x, min_y, max_x, max_y = [float(item) for item in target_geometry.bounds]
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)

    margin_lon = max(width * 0.55, 4.0)
    margin_lat = max(height * 0.55, 3.0)
    x0 = min_x - margin_lon
    x1 = max_x + margin_lon
    y0 = max(min_y - margin_lat, -90.0)
    y1 = min(max_y + margin_lat, 90.0)

    try:
        window = admin0_df.cx[x0:x1, y0:y1]
    except Exception:
        window = admin0_df

    context: list[Any] = []
    for _, row in window.iterrows():
        iso_raw = row.get(iso_col)
        iso3 = str(iso_raw).strip().upper() if iso_raw is not None else ""
        if iso3 == target_iso3:
            continue
        geometry = row.get("geometry")
        if _is_valid_geometry(geometry):
            context.append(geometry)
    return tuple(context)


def _partition_selected_cities(
    *,
    selected: SelectedCities,
    inset_geometries: Sequence[Any],
) -> tuple[
    tuple[tuple[CityRecord, bool], ...],
    tuple[tuple[tuple[CityRecord, bool], ...], ...],
]:
    groups: list[list[tuple[CityRecord, bool]]] = [[] for _ in inset_geometries]
    main: list[tuple[CityRecord, bool]] = []
    for city, is_capital in _selected_city_items(selected):
        inset_idx = _find_inset_index_for_city(city, inset_geometries)
        if inset_idx is None:
            main.append((city, is_capital))
        else:
            groups[inset_idx].append((city, is_capital))
    return (tuple(main), tuple(tuple(group) for group in groups))


def _selected_city_items(selected: SelectedCities) -> tuple[tuple[CityRecord, bool], ...]:
    items: list[tuple[CityRecord, bool]] = [(selected.capital, True)]
    items.extend((city, False) for city in selected.others)
    return tuple(items)


def _find_inset_index_for_city(city: CityRecord, inset_geometries: Sequence[Any]) -> int | None:
    if not inset_geometries:
        return None
    point = _city_to_point(city)
    for idx, geometry in enumerate(inset_geometries):
        if _geometry_matches_point(geometry, point):
            return idx
    return None


def _city_to_point(city: CityRecord) -> Any:
    return _require_shapely_point_factory()(city.lon, city.lat)


def _project_city_items(
    city_items: Sequence[tuple[CityRecord, bool]],
    projection: _LocalProjection,
) -> tuple[_ProjectedCity, ...]:
    out: list[_ProjectedCity] = []
    for city, is_capital in city_items:
        x, y = _project_lon_lat(city.lon, city.lat, projection)
        out.append(
            _ProjectedCity(
                name=city.name,
                source_name=city.name,
                x=x,
                y=y,
                is_capital=is_capital,
            )
        )
    return tuple(out)


def _is_valid_geometry(geometry: Any) -> bool:
    if geometry is None:
        return False
    if hasattr(geometry, "is_empty") and bool(geometry.is_empty):
        return False
    return True


def _build_local_projection(geometry: Any) -> _LocalProjection:
    _ = geometry
    return _LocalProjection(transformer=_require_pyproj_transformer())


def _project_geometry(geometry: Any, projection: _LocalProjection) -> Any:
    if not _is_valid_geometry(geometry):
        return geometry
    shapely_transform = _require_shapely_transform()
    return shapely_transform(projection.transformer.transform, geometry)


def _project_lon_lat(lon: float, lat: float, projection: _LocalProjection) -> tuple[float, float]:
    x, y = projection.transformer.transform(float(lon), float(lat))
    return (float(x), float(y))


def _compute_extent(
    *,
    geometry: Any,
    cfg: RenderConfig,
    projection: _LocalProjection,
    is_inset: bool,
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = [float(item) for item in geometry.bounds]
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    mode = _extent_mode_settings(is_inset=is_inset, cfg=cfg)

    configured_min_padding_m = max(cfg.extent.min_padding_deg * _EXTENT_POLICY.deg_to_m, 0.0)
    min_pad_x, min_pad_y = _adaptive_min_padding_m(
        width=width,
        height=height,
        configured_min_padding_m=configured_min_padding_m,
        is_inset=is_inset,
    )
    min_pad_x = max(min_pad_x, mode.min_padding_floor_m)
    min_pad_y = max(min_pad_y, mode.min_padding_floor_m)
    pad_x = max(width * cfg.extent.padding_ratio * mode.pad_scale, min_pad_x)
    pad_y = max(height * cfg.extent.padding_ratio * mode.pad_scale, min_pad_y)

    x0 = min_x - pad_x
    x1 = max_x + pad_x
    y0 = min_y - pad_y
    y1 = max_y + pad_y

    min_span_x = max(mode.min_span_m, width + pad_x * 2.0)
    min_span_y = max(mode.min_span_m, height + pad_y * 2.0)
    x0, x1 = _ensure_min_span(x0, x1, min_span_x)
    y0, y1 = _ensure_min_span(y0, y1, min_span_y)

    clamp_min = _mercator_y(cfg.extent.clamp_lat.min, projection)
    clamp_max = _mercator_y(cfg.extent.clamp_lat.max, projection)
    y0, y1 = _clamp_y_bounds(y0=y0, y1=y1, clamp_min=clamp_min, clamp_max=clamp_max)
    if y1 <= y0:
        center = max(min((min_y + max_y) / 2.0, clamp_max), clamp_min)
        y0 = max(center - 1_000.0, clamp_min)
        y1 = min(center + 1_000.0, clamp_max)
        if y1 <= y0:
            y1 = min(y0 + 1e-3, clamp_max)

    x0, x1, y0, y1 = _fit_extent_aspect(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        target_ratio=mode.target_ratio,
    )
    x0, x1, y0, y1 = _expand_extent_for_label_guard(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        is_inset=is_inset,
    )
    y0, y1 = _clamp_y_bounds(y0=y0, y1=y1, clamp_min=clamp_min, clamp_max=clamp_max)
    if y1 <= y0:
        y1 = min(y0 + 1e-3, clamp_max)
    return (x0, x1, y0, y1)


def _extent_mode_settings(*, is_inset: bool, cfg: RenderConfig) -> _ExtentModeSettings:
    if is_inset:
        return _ExtentModeSettings(
            min_padding_floor_m=_EXTENT_POLICY.inset_min_padding_floor_m,
            min_span_m=_EXTENT_POLICY.inset_min_span_m,
            pad_scale=_EXTENT_POLICY.inset_padding_scale,
            target_ratio=1.0,
        )
    return _ExtentModeSettings(
        min_padding_floor_m=_EXTENT_POLICY.main_min_padding_floor_m,
        min_span_m=_EXTENT_POLICY.main_min_span_m,
        pad_scale=1.0,
        target_ratio=cfg.image.width_px / cfg.image.height_px,
    )


def _clamp_y_bounds(
    *,
    y0: float,
    y1: float,
    clamp_min: float,
    clamp_max: float,
) -> tuple[float, float]:
    return (max(y0, clamp_min), min(y1, clamp_max))


def _compute_capital_centered_extent(
    *,
    capital: CityRecord,
    cfg: RenderConfig,
    projection: _LocalProjection,
    zoom_pct: float | None = None,
) -> tuple[float, float, float, float]:
    center_x, center_y = _project_lon_lat(capital.lon, capital.lat, projection)
    half_height = _EXTENT_POLICY.capital_center_half_height_m * _capital_zoom_scale(zoom_pct)
    target_ratio = cfg.image.width_px / cfg.image.height_px
    half_width = half_height * target_ratio
    x0 = center_x - half_width
    x1 = center_x + half_width
    y0 = center_y - half_height
    y1 = center_y + half_height
    x0, x1, y0, y1 = _expand_extent_for_label_guard(
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        is_inset=False,
    )
    clamp_min = _mercator_y(cfg.extent.clamp_lat.min, projection)
    clamp_max = _mercator_y(cfg.extent.clamp_lat.max, projection)
    y0, y1 = _clamp_y_bounds(y0=y0, y1=y1, clamp_min=clamp_min, clamp_max=clamp_max)
    if y1 <= y0:
        y1 = min(y0 + 1_000.0, clamp_max)
    return (x0, x1, y0, y1)


def _capital_zoom_scale(zoom_pct: float | None) -> float:
    if zoom_pct is None or abs(float(zoom_pct)) < 1e-9:
        return 1.0
    value = float(zoom_pct)
    if value > 0.0:
        # +100 => 2x zoom in => half spatial span.
        scale = 1.0 / (1.0 + value / 100.0)
    else:
        # -100 => 2x zoom out => double spatial span.
        scale = 1.0 + (abs(value) / 100.0)
    return max(0.05, min(scale, 20.0))


def _ensure_min_span(start: float, end: float, min_span: float) -> tuple[float, float]:
    span = max(end - start, 1e-6)
    if span >= min_span:
        return (start, end)
    center = (start + end) / 2.0
    half = min_span / 2.0
    return (center - half, center + half)


def _adaptive_min_padding_m(
    *,
    width: float,
    height: float,
    configured_min_padding_m: float,
    is_inset: bool,
) -> tuple[float, float]:
    if width <= _EXTENT_POLICY.tiny_country_max_m and height <= _EXTENT_POLICY.tiny_country_max_m:
        bias_x, bias_y = _tiny_padding_biases(is_inset=is_inset)
        adaptive_x = max(width * _EXTENT_POLICY.tiny_padding_x_scale + bias_x, 250.0)
        adaptive_y = max(height * _EXTENT_POLICY.tiny_padding_y_scale + bias_y, 250.0)
        return (
            min(configured_min_padding_m, adaptive_x),
            min(configured_min_padding_m, adaptive_y),
        )
    return (configured_min_padding_m, configured_min_padding_m)


def _tiny_padding_biases(*, is_inset: bool) -> tuple[float, float]:
    if is_inset:
        return (
            _EXTENT_POLICY.tiny_padding_x_bias_inset,
            _EXTENT_POLICY.tiny_padding_y_bias_inset,
        )
    return (
        _EXTENT_POLICY.tiny_padding_x_bias_main,
        _EXTENT_POLICY.tiny_padding_y_bias_main,
    )


def _expand_extent_for_label_guard(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    is_inset: bool,
) -> tuple[float, float, float, float]:
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    min_guard = _LABEL_POLICY.edge_guard_min_inset_m if is_inset else _LABEL_POLICY.edge_guard_min_main_m
    guard_x = max(width * _LABEL_POLICY.edge_guard_ratio, min_guard)
    guard_y = max(height * _LABEL_POLICY.edge_guard_ratio, min_guard)
    return (x0 - guard_x, x1 + guard_x, y0 - guard_y, y1 + guard_y)


def _fit_extent_aspect(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    target_ratio: float,
) -> tuple[float, float, float, float]:
    width = max(x1 - x0, 1e-6)
    height = max(y1 - y0, 1e-6)
    current_ratio = width / height

    if current_ratio < target_ratio:
        needed_width = target_ratio * height
        expand = max((needed_width - width) / 2.0, 0.0)
        x0 -= expand
        x1 += expand
        return (x0, x1, y0, y1)

    needed_height = width / target_ratio
    expand = max((needed_height - height) / 2.0, 0.0)
    y0 -= expand
    y1 += expand
    return (x0, x1, y0, y1)


def _draw_context_outlines(*, ax: Any, geometries: Sequence[Any]) -> None:
    for geometry in geometries:
        if not _is_valid_geometry(geometry):
            continue
        _draw_geometry_outline(
            ax=ax,
            geometry=geometry,
            color=_GEOMETRY_DRAW_POLICY.context_line_color,
            line_width=0.8,
            alpha=_GEOMETRY_DRAW_POLICY.context_line_alpha,
            line_style=_GEOMETRY_DRAW_POLICY.context_line_style,
            zorder=0,
        )


def _draw_geometry_outline(
    *,
    ax: Any,
    geometry: Any,
    color: str,
    line_width: float,
    alpha: float = 1.0,
    line_style: str | tuple[Any, ...] = "solid",
    zorder: int = 1,
) -> None:
    for ring in _iter_linear_rings(geometry):
        if len(ring) < 2:
            continue
        for segment in _split_ring_segments(ring):
            if len(segment) < 2:
                continue
            x_values = [float(point[0]) for point in segment]
            y_values = [float(point[1]) for point in segment]
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=line_width,
                alpha=alpha,
                linestyle=line_style,
                zorder=zorder,
                solid_joinstyle="round",
                solid_capstyle="round",
            )


def _split_ring_segments(
    ring: Sequence[tuple[float, float]],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if len(ring) < 2:
        return ()
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [ring[0]]
    for point in ring[1:]:
        prev = current[-1]
        if _is_large_segment_jump(prev, point):
            if len(current) >= 2:
                segments.append(current)
            current = [point]
            continue
        current.append(point)
    if len(current) >= 2:
        segments.append(current)
    return tuple(tuple(segment) for segment in segments)


def _is_large_segment_jump(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    dx = abs(float(right[0]) - float(left[0]))
    dy = abs(float(right[1]) - float(left[1]))
    if dx > _GEOMETRY_DRAW_POLICY.segment_jump_threshold_m:
        return True
    if math.hypot(dx, dy) > _GEOMETRY_DRAW_POLICY.segment_jump_threshold_m * 1.1:
        return True
    return False


def _iter_linear_rings(geometry: Any) -> Sequence[Sequence[tuple[float, float]]]:
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type == "Polygon":
        exterior = [(float(x), float(y)) for x, y in geometry.exterior.coords]
        rings: list[Sequence[tuple[float, float]]] = [exterior]
        for interior in geometry.interiors:
            rings.append([(float(x), float(y)) for x, y in interior.coords])
        return rings

    if geom_type == "MultiPolygon":
        rings: list[Sequence[tuple[float, float]]] = []
        for polygon in geometry.geoms:
            rings.extend(_iter_linear_rings(polygon))
        return rings

    if geom_type == "GeometryCollection":
        rings: list[Sequence[tuple[float, float]]] = []
        for part in geometry.geoms:
            rings.extend(_iter_linear_rings(part))
        return rings

    return []


def _draw_city_markers(
    *,
    ax: Any,
    cities: Sequence[_ProjectedCity],
    marker_color: str,
    city_marker_size: int,
    capital_marker_size: int,
    zorder: int,
) -> None:
    normal = [city for city in cities if not city.is_capital]
    capitals = [city for city in cities if city.is_capital]
    city_size = max(
        float(city_marker_size) * 1.4 * _LABEL_POLICY.city_visual_scale,
        18.0 * _LABEL_POLICY.city_visual_scale,
    )
    capital_size = max(
        float(capital_marker_size) * 2.0 * _LABEL_POLICY.city_visual_scale,
        95.0 * _LABEL_POLICY.city_visual_scale,
    )
    if normal:
        ax.scatter(
            [city.x for city in normal],
            [city.y for city in normal],
            s=[city_size] * len(normal),
            c=marker_color,
            marker="o",
            linewidths=0.0,
            zorder=zorder,
        )
    if capitals:
        ax.scatter(
            [city.x for city in capitals],
            [city.y for city in capitals],
            s=[capital_size] * len(capitals),
            c=marker_color,
            marker="*",
            linewidths=0.0,
            zorder=zorder + 1,
        )


def _place_city_labels(
    *,
    ax: Any,
    fig: Any,
    transforms: Any,
    cities: Sequence[_ProjectedCity],
    label_overrides: Mapping[str, str],
    cfg: RenderConfig,
    zorder_base: int,
) -> None:
    candidates = _build_label_candidates(cities=cities, label_overrides=label_overrides)
    if not candidates:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied: list[_PixelBBox] = []
    for candidate in candidates:
        placement = _place_label_candidate(
            ax=ax,
            fig=fig,
            renderer=renderer,
            transforms=transforms,
            candidate=candidate,
            occupied=occupied,
            cfg=cfg,
            zorder_base=zorder_base,
        )
        if placement is not None:
            occupied.append(placement)


def _build_label_candidates(
    *,
    cities: Sequence[_ProjectedCity],
    label_overrides: Mapping[str, str],
) -> tuple[_LabelCandidate, ...]:
    ordered = sorted(cities, key=lambda city: (not city.is_capital, city.source_name.casefold()))
    out: list[_LabelCandidate] = []
    for city in ordered:
        label = label_overrides.get(city.source_name, city.name)
        if not label:
            continue
        out.append(_LabelCandidate(city=city, text=label))
    return tuple(out)


def _place_label_candidate(
    *,
    ax: Any,
    fig: Any,
    renderer: Any,
    transforms: Any,
    candidate: _LabelCandidate,
    occupied: Sequence[_PixelBBox],
    cfg: RenderConfig,
    zorder_base: int,
) -> _PixelBBox | None:
    best: tuple[float, Any, _PixelBBox] | None = None
    for dx_px, dy_px in cfg.labels.offsets_px:
        artist = _create_label_artist(
            ax=ax,
            fig=fig,
            transforms=transforms,
            candidate=candidate,
            dx_px=dx_px,
            dy_px=dy_px,
            cfg=cfg,
            zorder_base=zorder_base,
        )
        bbox = _expanded_text_bbox(
            artist=artist,
            renderer=renderer,
            padding_px=cfg.labels.collision_padding_px,
        )
        overlap = _total_overlap_area(bbox, occupied)
        if overlap <= 0.0:
            if best is not None:
                best[1].remove()
            return bbox
        if candidate.city.is_capital and cfg.labels.allow_capital_overlap_if_needed:
            if best is None or overlap < best[0]:
                if best is not None:
                    best[1].remove()
                best = (overlap, artist, bbox)
                continue
        artist.remove()

    if best is not None:
        return best[2]
    return None


def _create_label_artist(
    *,
    ax: Any,
    fig: Any,
    transforms: Any,
    candidate: _LabelCandidate,
    dx_px: int,
    dy_px: int,
    cfg: RenderConfig,
    zorder_base: int,
) -> Any:
    if dx_px > 0:
        ha = "left"
    elif dx_px < 0:
        ha = "right"
    else:
        ha = "center"
    if dy_px > 0:
        va = "bottom"
    elif dy_px < 0:
        va = "top"
    else:
        va = "center"

    shift = transforms.ScaledTranslation(dx_px / fig.dpi, dy_px / fig.dpi, fig.dpi_scale_trans)
    transform = ax.transData + shift
    city = candidate.city
    font_size = cfg.style.font_size_capital if city.is_capital else cfg.style.font_size_city
    font_size *= _LABEL_POLICY.city_visual_scale
    font_weight = cfg.style.font_weight_capital if city.is_capital else "normal"
    zorder = zorder_base + 1 if city.is_capital else zorder_base
    return ax.text(
        city.x,
        city.y,
        candidate.text,
        transform=transform,
        color=cfg.style.label_color,
        fontsize=font_size,
        fontweight=font_weight,
        family=cfg.style.font_family,
        ha=ha,
        va=va,
        clip_on=False,
        zorder=zorder,
    )


def _expanded_text_bbox(
    *,
    artist: Any,
    renderer: Any,
    padding_px: int,
) -> _PixelBBox:
    bbox = artist.get_window_extent(renderer=renderer)
    return (
        float(bbox.x0) - padding_px,
        float(bbox.y0) - padding_px,
        float(bbox.x1) + padding_px,
        float(bbox.y1) + padding_px,
    )


def _total_overlap_area(
    bbox: _PixelBBox,
    occupied: Sequence[_PixelBBox],
) -> float:
    return sum(_intersection_area(bbox, current) for current in occupied)


def _intersection_area(
    left: _PixelBBox,
    right: _PixelBBox,
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _inset_positions(count: int) -> tuple[tuple[float, float, float, float], ...]:
    if count < 1:
        return ()
    available = max(1.0 - (2.0 * _INSET_POLICY.box_margin), 0.1)
    max_by_width = (
        available - _INSET_POLICY.box_gap * max(count - 1, 0)
    ) / max(count, 1)
    size = max(min(_INSET_POLICY.box_size, max_by_width), 0.08)
    y = 1.0 - _INSET_POLICY.box_margin - size

    out: list[tuple[float, float, float, float]] = []
    for idx in range(count):
        x = _INSET_POLICY.box_margin + idx * (size + _INSET_POLICY.box_gap)
        out.append((x, y, size, size))
    return tuple(out)


def _apply_background(*, fig: Any | None, ax: Any, background: str) -> None:
    if background.casefold() == "transparent":
        if fig is not None:
            fig.patch.set_facecolor("white")
            fig.patch.set_alpha(0.0)
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    else:
        if fig is not None:
            fig.patch.set_facecolor(background)
        ax.set_facecolor(background)


def _resolve_basemap_source(mode: str) -> Any | None:
    chosen = mode.casefold()
    if chosen == _BACKGROUND_WHITE:
        return None
    providers = _require_xyzservices_providers()
    if chosen == _BACKGROUND_SATELLITE:
        return providers.Esri.WorldImagery
    # default flat map with labels removed
    return providers.CartoDB.PositronNoLabels


def _mercator_y(lat: float, projection: _LocalProjection) -> float:
    _, y = projection.transformer.transform(0.0, float(lat))
    return float(y)


def _require_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import matplotlib.transforms as transforms
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for map rendering") from exc
    return (plt, transforms)


@lru_cache(maxsize=1)
def _require_contextily() -> Any:
    try:
        import contextily as ctx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("contextily is required for real basemap rendering") from exc
    return ctx


@lru_cache(maxsize=1)
def _require_xyzservices_providers() -> Any:
    try:
        from xyzservices import providers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xyzservices is required for basemap source definitions") from exc
    return providers


def _require_shapely_transform() -> Any:
    try:
        from shapely.ops import transform
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("shapely is required for geometry projection in rendering") from exc
    return transform


def _require_shapely_unary_union() -> Any:
    try:
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("shapely is required for geometry union in rendering") from exc
    return unary_union


def _require_shapely_point_factory() -> Any:
    try:
        from shapely.geometry import Point
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("shapely is required for point checks in rendering") from exc
    return Point


def _require_shapely_box_factory() -> Any:
    try:
        from shapely.geometry import box
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("shapely is required for viewport checks in rendering") from exc
    return box


@lru_cache(maxsize=1)
def _require_pyproj_transformer() -> Any:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyproj is required for Web Mercator projection in rendering") from exc
    return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _format_code_list(values: list[str], limit: int = 12) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    shown = ", ".join(values[:limit])
    return f"{shown}, ... (+{len(values) - limit} more)"
