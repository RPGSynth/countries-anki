"""Typed configuration loader for `config.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import yaml


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for '{field_name}'")
    return cast(Mapping[str, Any], value)


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for '{field_name}'")
    return value.strip()


def _int(value: Any, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for '{field_name}'")
    return value


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected float for '{field_name}'")


def _bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Expected bool for '{field_name}'")
    return value


def _str_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list for '{field_name}'")
    out: list[str] = []
    for idx, item in enumerate(value):
        out.append(_str(item, f"{field_name}[{idx}]"))
    return tuple(out)


def _path_from_cfg(value: Any, field_name: str, root_dir: Path) -> Path:
    raw = _str(value, field_name)
    p = Path(raw)
    return p if p.is_absolute() else root_dir / p


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    deck_name: str
    deck_id: int
    model_id: int
    output_apkg: Path
    language: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], root_dir: Path) -> ProjectConfig:
        return cls(
            deck_name=_str(raw.get("deck_name"), "project.deck_name"),
            deck_id=_int(raw.get("deck_id"), "project.deck_id"),
            model_id=_int(raw.get("model_id"), "project.model_id"),
            output_apkg=_path_from_cfg(raw.get("output_apkg"), "project.output_apkg", root_dir),
            language=_str(raw.get("language"), "project.language"),
        )


@dataclass(frozen=True, slots=True)
class PathsConfig:
    ne_admin0_countries: Path
    ne_populated_places: Path
    un_members: Path
    cities_overrides: Path
    flags_overrides: Path
    render_overrides: Path
    build_root: Path
    maps_dir: Path
    flags_dir: Path
    qa_dir: Path
    attribution_dir: Path
    manifests_dir: Path
    logs_dir: Path

    @property
    def required_input_files(self) -> tuple[Path, ...]:
        return (
            self.un_members,
            self.cities_overrides,
            self.flags_overrides,
            self.render_overrides,
        )

    @property
    def build_directories(self) -> tuple[Path, ...]:
        return (
            self.build_root,
            self.maps_dir,
            self.flags_dir,
            self.qa_dir,
            self.attribution_dir,
            self.manifests_dir,
            self.logs_dir,
            self.qa_dir / "thumbs",
            self.build_root / "media",
            self.build_root / "deck",
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], root_dir: Path) -> PathsConfig:
        return cls(
            ne_admin0_countries=_path_from_cfg(
                raw.get("ne_admin0_countries"), "paths.ne_admin0_countries", root_dir
            ),
            ne_populated_places=_path_from_cfg(
                raw.get("ne_populated_places"), "paths.ne_populated_places", root_dir
            ),
            un_members=_path_from_cfg(raw.get("un_members"), "paths.un_members", root_dir),
            cities_overrides=_path_from_cfg(
                raw.get("cities_overrides"), "paths.cities_overrides", root_dir
            ),
            flags_overrides=_path_from_cfg(
                raw.get("flags_overrides"), "paths.flags_overrides", root_dir
            ),
            render_overrides=_path_from_cfg(
                raw.get("render_overrides"), "paths.render_overrides", root_dir
            ),
            build_root=_path_from_cfg(raw.get("build_root"), "paths.build_root", root_dir),
            maps_dir=_path_from_cfg(raw.get("maps_dir"), "paths.maps_dir", root_dir),
            flags_dir=_path_from_cfg(raw.get("flags_dir"), "paths.flags_dir", root_dir),
            qa_dir=_path_from_cfg(raw.get("qa_dir"), "paths.qa_dir", root_dir),
            attribution_dir=_path_from_cfg(
                raw.get("attribution_dir"), "paths.attribution_dir", root_dir
            ),
            manifests_dir=_path_from_cfg(raw.get("manifests_dir"), "paths.manifests_dir", root_dir),
            logs_dir=_path_from_cfg(raw.get("logs_dir"), "paths.logs_dir", root_dir),
        )


@dataclass(frozen=True, slots=True)
class CountryPolicyConfig:
    include_un_members_only: bool
    strict: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CountryPolicyConfig:
        return cls(
            include_un_members_only=_bool(
                raw.get("include_un_members_only"), "country_policy.include_un_members_only"
            ),
            strict=_bool(raw.get("strict"), "country_policy.strict"),
        )


@dataclass(frozen=True, slots=True)
class CitiesSortConfig:
    primary: str
    secondary: str
    tiebreaker: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CitiesSortConfig:
        return cls(
            primary=_str(raw.get("primary"), "cities.sort.primary"),
            secondary=_str(raw.get("secondary"), "cities.sort.secondary"),
            tiebreaker=_str(raw.get("tiebreaker"), "cities.sort.tiebreaker"),
        )


@dataclass(frozen=True, slots=True)
class CapitalChoiceConfig:
    prefer_low_scalerank: bool
    prefer_high_pop: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CapitalChoiceConfig:
        return cls(
            prefer_low_scalerank=_bool(
                raw.get("prefer_low_scalerank"), "cities.capital_choice.prefer_low_scalerank"
            ),
            prefer_high_pop=_bool(raw.get("prefer_high_pop"), "cities.capital_choice.prefer_high_pop"),
        )


@dataclass(frozen=True, slots=True)
class TinyCountryBBoxConfig:
    width: float
    height: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TinyCountryBBoxConfig:
        return cls(
            width=_float(raw.get("width"), "cities.tiny_country_bbox_deg.width"),
            height=_float(raw.get("height"), "cities.tiny_country_bbox_deg.height"),
        )


@dataclass(frozen=True, slots=True)
class CitiesConfig:
    default_n: int
    min_n_for_tiny: int
    tiny_country_bbox_deg: TinyCountryBBoxConfig
    tiny_country_area_deg2: float
    sort: CitiesSortConfig
    capital_fields: tuple[str, ...]
    capital_choice: CapitalChoiceConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CitiesConfig:
        default_n = _int(raw.get("default_n"), "cities.default_n")
        min_n_for_tiny = _int(raw.get("min_n_for_tiny"), "cities.min_n_for_tiny")
        if default_n < 0:
            raise ValueError("cities.default_n must be >= 0")
        if min_n_for_tiny < 0:
            raise ValueError("cities.min_n_for_tiny must be >= 0")
        if min_n_for_tiny > default_n:
            raise ValueError("cities.min_n_for_tiny cannot be greater than cities.default_n")

        return cls(
            default_n=default_n,
            min_n_for_tiny=min_n_for_tiny,
            tiny_country_bbox_deg=TinyCountryBBoxConfig.from_mapping(
                _mapping(raw.get("tiny_country_bbox_deg"), "cities.tiny_country_bbox_deg")
            ),
            tiny_country_area_deg2=_float(raw.get("tiny_country_area_deg2"), "cities.tiny_country_area_deg2"),
            sort=CitiesSortConfig.from_mapping(_mapping(raw.get("sort"), "cities.sort")),
            capital_fields=_str_list(raw.get("capital_fields"), "cities.capital_fields"),
            capital_choice=CapitalChoiceConfig.from_mapping(
                _mapping(raw.get("capital_choice"), "cities.capital_choice")
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderImageConfig:
    width_px: int
    height_px: int
    dpi: int
    background: str
    format: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RenderImageConfig:
        return cls(
            width_px=_int(raw.get("width_px"), "render.image.width_px"),
            height_px=_int(raw.get("height_px"), "render.image.height_px"),
            dpi=_int(raw.get("dpi"), "render.image.dpi"),
            background=_str(raw.get("background"), "render.image.background"),
            format=_str(raw.get("format"), "render.image.format"),
        )


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    crs: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ProjectionConfig:
        return cls(crs=_str(raw.get("crs"), "render.projection.crs"))


@dataclass(frozen=True, slots=True)
class ClampLatConfig:
    min: float
    max: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ClampLatConfig:
        return cls(
            min=_float(raw.get("min"), "render.extent.clamp_lat.min"),
            max=_float(raw.get("max"), "render.extent.clamp_lat.max"),
        )


@dataclass(frozen=True, slots=True)
class RenderExtentConfig:
    padding_ratio: float
    min_padding_deg: float
    clamp_lat: ClampLatConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RenderExtentConfig:
        return cls(
            padding_ratio=_float(raw.get("padding_ratio"), "render.extent.padding_ratio"),
            min_padding_deg=_float(raw.get("min_padding_deg"), "render.extent.min_padding_deg"),
            clamp_lat=ClampLatConfig.from_mapping(_mapping(raw.get("clamp_lat"), "render.extent.clamp_lat")),
        )


@dataclass(frozen=True, slots=True)
class RenderBackgroundConfig:
    mode: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RenderBackgroundConfig:
        mode = _str(raw.get("mode"), "render.background.mode").casefold()
        if mode == "flat":
            mode = "google_flat"
        allowed = {"white", "google_flat", "satellite"}
        if mode not in allowed:
            raise ValueError(
                "render.background.mode must be one of: "
                + ", ".join(sorted(allowed))
            )
        return cls(mode=mode)

    @classmethod
    def default(cls) -> RenderBackgroundConfig:
        return cls(mode="google_flat")


@dataclass(frozen=True, slots=True)
class RenderStyleConfig:
    country_outline_width: float
    country_outline_color: str
    coastline: bool
    city_marker_color: str
    city_marker_size: int
    capital_marker_size: int
    label_color: str
    font_family: str
    font_size_city: int
    font_size_capital: int
    font_weight_capital: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RenderStyleConfig:
        return cls(
            country_outline_width=_float(raw.get("country_outline_width"), "render.style.country_outline_width"),
            country_outline_color=_str(raw.get("country_outline_color"), "render.style.country_outline_color"),
            coastline=_bool(raw.get("coastline"), "render.style.coastline"),
            city_marker_color=_str(raw.get("city_marker_color"), "render.style.city_marker_color"),
            city_marker_size=_int(raw.get("city_marker_size"), "render.style.city_marker_size"),
            capital_marker_size=_int(raw.get("capital_marker_size"), "render.style.capital_marker_size"),
            label_color=_str(raw.get("label_color"), "render.style.label_color"),
            font_family=_str(raw.get("font_family"), "render.style.font_family"),
            font_size_city=_int(raw.get("font_size_city"), "render.style.font_size_city"),
            font_size_capital=_int(raw.get("font_size_capital"), "render.style.font_size_capital"),
            font_weight_capital=_str(raw.get("font_weight_capital"), "render.style.font_weight_capital"),
        )


@dataclass(frozen=True, slots=True)
class LabelsConfig:
    offsets_px: tuple[tuple[int, int], ...]
    allow_capital_overlap_if_needed: bool
    collision_padding_px: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> LabelsConfig:
        offsets_raw = raw.get("offsets_px")
        if not isinstance(offsets_raw, list) or not offsets_raw:
            raise ValueError("Expected non-empty list for 'render.labels.offsets_px'")
        offsets: list[tuple[int, int]] = []
        for idx, item in enumerate(offsets_raw):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"Invalid render.labels.offsets_px[{idx}]")
            dx = _int(item[0], f"render.labels.offsets_px[{idx}][0]")
            dy = _int(item[1], f"render.labels.offsets_px[{idx}][1]")
            offsets.append((dx, dy))
        return cls(
            offsets_px=tuple(offsets),
            allow_capital_overlap_if_needed=_bool(
                raw.get("allow_capital_overlap_if_needed"),
                "render.labels.allow_capital_overlap_if_needed",
            ),
            collision_padding_px=_int(raw.get("collision_padding_px"), "render.labels.collision_padding_px"),
        )


@dataclass(frozen=True, slots=True)
class RenderConfig:
    image: RenderImageConfig
    projection: ProjectionConfig
    extent: RenderExtentConfig
    background: RenderBackgroundConfig
    style: RenderStyleConfig
    labels: LabelsConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RenderConfig:
        background_raw = raw.get("background")
        background = (
            RenderBackgroundConfig.default()
            if background_raw is None
            else RenderBackgroundConfig.from_mapping(_mapping(background_raw, "render.background"))
        )
        return cls(
            image=RenderImageConfig.from_mapping(_mapping(raw.get("image"), "render.image")),
            projection=ProjectionConfig.from_mapping(
                _mapping(raw.get("projection"), "render.projection")
            ),
            extent=RenderExtentConfig.from_mapping(_mapping(raw.get("extent"), "render.extent")),
            background=background,
            style=RenderStyleConfig.from_mapping(_mapping(raw.get("style"), "render.style")),
            labels=LabelsConfig.from_mapping(_mapping(raw.get("labels"), "render.labels")),
        )


@dataclass(frozen=True, slots=True)
class FlagOutputConfig:
    height_px: int
    background: str
    padding_px: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FlagOutputConfig:
        return cls(
            height_px=_int(raw.get("height_px"), "flags.output.height_px"),
            background=_str(raw.get("background"), "flags.output.background"),
            padding_px=_int(raw.get("padding_px"), "flags.output.padding_px"),
        )


@dataclass(frozen=True, slots=True)
class FlagSvgConfig:
    prefer_svg: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FlagSvgConfig:
        return cls(prefer_svg=_bool(raw.get("prefer_svg"), "flags.svg.prefer_svg"))


@dataclass(frozen=True, slots=True)
class FlagsConfig:
    source: str
    cache_http: bool
    request_timeout_s: int
    user_agent: str
    min_request_interval_s: float
    max_retries: int
    retry_backoff_s: float
    output: FlagOutputConfig
    svg: FlagSvgConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FlagsConfig:
        min_request_interval_s = _float(
            raw.get("min_request_interval_s", 0.8),
            "flags.min_request_interval_s",
        )
        max_retries = _int(raw.get("max_retries", 5), "flags.max_retries")
        retry_backoff_s = _float(raw.get("retry_backoff_s", 1.0), "flags.retry_backoff_s")
        if min_request_interval_s < 0:
            raise ValueError("flags.min_request_interval_s must be >= 0")
        if max_retries < 0:
            raise ValueError("flags.max_retries must be >= 0")
        if retry_backoff_s <= 0:
            raise ValueError("flags.retry_backoff_s must be > 0")

        return cls(
            source=_str(raw.get("source"), "flags.source"),
            cache_http=_bool(raw.get("cache_http"), "flags.cache_http"),
            request_timeout_s=_int(raw.get("request_timeout_s"), "flags.request_timeout_s"),
            user_agent=_str(raw.get("user_agent"), "flags.user_agent"),
            min_request_interval_s=min_request_interval_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            output=FlagOutputConfig.from_mapping(_mapping(raw.get("output"), "flags.output")),
            svg=FlagSvgConfig.from_mapping(_mapping(raw.get("svg"), "flags.svg")),
        )


@dataclass(frozen=True, slots=True)
class AnkiTemplatesConfig:
    front_title_size_px: int
    back_title_size_px: int
    map_max_width_pct: int
    flag_max_height_px: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AnkiTemplatesConfig:
        return cls(
            front_title_size_px=_int(raw.get("front_title_size_px"), "anki.templates.front_title_size_px"),
            back_title_size_px=_int(raw.get("back_title_size_px"), "anki.templates.back_title_size_px"),
            map_max_width_pct=_int(raw.get("map_max_width_pct"), "anki.templates.map_max_width_pct"),
            flag_max_height_px=_int(raw.get("flag_max_height_px"), "anki.templates.flag_max_height_px"),
        )


@dataclass(frozen=True, slots=True)
class AnkiConfig:
    tags: tuple[str, ...]
    templates: AnkiTemplatesConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AnkiConfig:
        return cls(
            tags=_str_list(raw.get("tags"), "anki.tags"),
            templates=AnkiTemplatesConfig.from_mapping(
                _mapping(raw.get("templates"), "anki.templates")
            ),
        )


@dataclass(frozen=True, slots=True)
class QaConfig:
    generate_index: bool
    thumbnail_width_px: int
    max_columns: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> QaConfig:
        return cls(
            generate_index=_bool(raw.get("generate_index"), "qa.generate_index"),
            thumbnail_width_px=_int(raw.get("thumbnail_width_px"), "qa.thumbnail_width_px"),
            max_columns=_int(raw.get("max_columns"), "qa.max_columns"),
        )


@dataclass(frozen=True, slots=True)
class BuildConfig:
    write_manifest: bool
    manifest_include_hashes: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BuildConfig:
        return cls(
            write_manifest=_bool(raw.get("write_manifest"), "build.write_manifest"),
            manifest_include_hashes=_bool(
                raw.get("manifest_include_hashes"), "build.manifest_include_hashes"
            ),
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    source_path: Path
    project: ProjectConfig
    paths: PathsConfig
    country_policy: CountryPolicyConfig
    cities: CitiesConfig
    render: RenderConfig
    flags: FlagsConfig
    anki: AnkiConfig
    qa: QaConfig
    build: BuildConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], source_path: Path) -> AppConfig:
        root_dir = source_path.parent.resolve()
        return cls(
            source_path=source_path.resolve(),
            project=ProjectConfig.from_mapping(_mapping(raw.get("project"), "project"), root_dir),
            paths=PathsConfig.from_mapping(_mapping(raw.get("paths"), "paths"), root_dir),
            country_policy=CountryPolicyConfig.from_mapping(
                _mapping(raw.get("country_policy"), "country_policy")
            ),
            cities=CitiesConfig.from_mapping(_mapping(raw.get("cities"), "cities")),
            render=RenderConfig.from_mapping(_mapping(raw.get("render"), "render")),
            flags=FlagsConfig.from_mapping(_mapping(raw.get("flags"), "flags")),
            anki=AnkiConfig.from_mapping(_mapping(raw.get("anki"), "anki")),
            qa=QaConfig.from_mapping(_mapping(raw.get("qa"), "qa")),
            build=BuildConfig.from_mapping(_mapping(raw.get("build"), "build")),
        )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the YAML config file into typed settings."""
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError("Top-level config must be a YAML mapping")
    return AppConfig.from_mapping(cast(Mapping[str, Any], raw), cfg_path)
