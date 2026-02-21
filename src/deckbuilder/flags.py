"""Wikimedia flag retrieval and normalization pipeline."""

from __future__ import annotations

import html
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote

import requests
import yaml
from PIL import Image

from .config import AppConfig, FlagsConfig
from .countries import load_un_members
from .models import CountrySpec, FlagOverride
from .util import write_json


_WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
_COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_RETRYABLE_HTTP_STATUS = {403, 429, 500, 502, 503, 504}

_LOGGER = logging.getLogger("deckbuilder.flags")


@dataclass(frozen=True, slots=True)
class FlagAttribution:
    iso3: str
    commons_title: str
    source_url: str
    license_name: str
    author: str | None
    retrieved_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "iso3": self.iso3,
            "commons_title": self.commons_title,
            "source_url": self.source_url,
            "license_name": self.license_name,
            "author": self.author,
            "retrieved_at_utc": self.retrieved_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FlagAttribution | None:
        iso3 = raw.get("iso3")
        title = raw.get("commons_title")
        source_url = raw.get("source_url")
        license_name = raw.get("license_name")
        retrieved = raw.get("retrieved_at_utc")
        required = (iso3, title, source_url, license_name, retrieved)
        if not all(isinstance(item, str) and item.strip() for item in required):
            return None
        author_raw = raw.get("author")
        author = author_raw if isinstance(author_raw, str) and author_raw.strip() else None
        return cls(
            iso3=iso3.strip().upper(),
            commons_title=title.strip(),
            source_url=source_url.strip(),
            license_name=license_name.strip(),
            author=author,
            retrieved_at_utc=retrieved.strip(),
        )


@dataclass(slots=True)
class FlagFetchReport:
    output_dir: Path | None = None
    attribution_path: Path | None = None
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


@dataclass(frozen=True, slots=True)
class _CommonsImageInfo:
    canonical_title: str
    source_url: str
    thumb_url: str | None
    mime: str | None
    extmetadata: Mapping[str, Any]


def load_flags_overrides(path: Path) -> dict[str, FlagOverride]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")

    overrides: dict[str, FlagOverride] = {}
    for iso3_raw, value in raw.items():
        if not isinstance(iso3_raw, str):
            raise ValueError("flags override keys must be ISO3 strings")
        iso3 = iso3_raw.strip().upper()
        if len(iso3) != 3:
            raise ValueError(f"Invalid ISO3 override key: {iso3_raw}")
        if not isinstance(value, dict):
            raise ValueError(f"flags override value for {iso3} must be a mapping")
        overrides[iso3] = FlagOverride.from_mapping(value)
    return overrides


def run_fetch_flags(
    cfg: AppConfig,
    *,
    country_filter: Sequence[str] | None = None,
    limit_countries: int | None = None,
    clean_flags: bool = False,
) -> FlagFetchReport:
    report = FlagFetchReport(
        output_dir=cfg.paths.flags_dir,
        attribution_path=cfg.paths.attribution_dir / "flags.json",
    )
    countries = load_un_members(cfg.paths.un_members)
    if not countries:
        report.add_error("UN members list is empty; cannot fetch flags.")
        return report
    countries = sorted(countries, key=lambda item: item.iso3)

    if limit_countries is not None and limit_countries < 1:
        report.add_error("limit_countries must be >= 1 when provided.")
        return report

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
        report.add_info(f"Country fetch limit enabled: first {len(countries)} countries.")

    if not countries:
        report.add_error("No countries selected for flag fetching after filters/limits.")
        return report

    if clean_flags:
        removed = 0
        for path in cfg.paths.flags_dir.glob("flag_*.png"):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                report.add_warning(f"Failed removing old flag file {path}: {exc}")
        report.add_info(f"Cleaned {removed} existing flag PNG files from {cfg.paths.flags_dir}")

    try:
        overrides = load_flags_overrides(cfg.paths.flags_overrides)
    except Exception as exc:
        report.add_error(f"Failed loading flags overrides '{cfg.paths.flags_overrides}': {exc}")
        return report
    report.add_info(f"Loaded {len(overrides)} flag override entries")

    existing_attributions = _load_existing_attributions(report.attribution_path)
    fetcher = WikimediaFlagFetcher(cfg.flags)

    results: dict[str, FlagAttribution] = {}
    fetched = 0
    reused = 0
    failed: list[str] = []

    for idx, country in enumerate(countries, start=1):
        output_path = cfg.paths.flags_dir / f"flag_{country.iso3}.png"
        cached_attr = existing_attributions.get(country.iso3)
        if cfg.flags.cache_http and output_path.exists() and cached_attr is not None:
            results[country.iso3] = cached_attr
            reused += 1
            _LOGGER.info("[flags] (%d/%d) reused %s", idx, len(countries), country.iso3)
            continue

        try:
            attr = fetcher.fetch_and_normalize(
                country=country,
                output_path=output_path,
                override=overrides.get(country.iso3),
            )
        except Exception as exc:
            failed.append(f"{country.iso3}({exc})")
            _LOGGER.error("[flags] (%d/%d) failed %s: %s", idx, len(countries), country.iso3, exc)
            continue

        results[country.iso3] = attr
        fetched += 1
        _LOGGER.info("[flags] (%d/%d) fetched %s", idx, len(countries), country.iso3)

    if report.attribution_path is not None:
        payload = [results[iso3].to_dict() for iso3 in sorted(results)]
        write_json(report.attribution_path, payload)
        report.add_info(f"Flag attribution written to {report.attribution_path}")

    report.summary = {
        "countries_total": len(countries),
        "flags_fetched": fetched,
        "flags_reused": reused,
        "flags_failed": len(failed),
        "attributions_written": len(results),
    }
    report.add_info(
        "Flag fetch summary: "
        f"countries_total={len(countries)}, "
        f"flags_fetched={fetched}, "
        f"flags_reused={reused}, "
        f"flags_failed={len(failed)}"
    )

    if failed:
        report.add_error("Flag fetch failures: " + _format_code_list(sorted(failed)))
    elif report.output_dir is not None:
        report.add_info(f"Flag files written to {report.output_dir}")
    return report


def format_flag_lines(report: FlagFetchReport) -> Sequence[str]:
    lines: list[str] = []
    lines.extend(f"[INFO] {msg}" for msg in report.infos)
    lines.extend(f"[WARN] {msg}" for msg in report.warnings)
    lines.extend(f"[ERROR] {msg}" for msg in report.errors)
    if report.ok:
        lines.append("[OK] Flag fetching completed with no errors.")
    return lines


class WikimediaFlagFetcher:
    """Fetch Wikimedia flag images and normalize them into stable PNG outputs."""

    def __init__(self, cfg: FlagsConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": cfg.user_agent})
        self._min_request_interval_s = max(float(cfg.min_request_interval_s), 0.0)
        self._max_retries = max(int(cfg.max_retries), 0)
        self._retry_backoff_s = max(float(cfg.retry_backoff_s), 0.01)
        self._last_request_started_at: float | None = None
        self._iso3_to_title: dict[str, str] | None = None

    def fetch_and_normalize(
        self,
        *,
        country: CountrySpec,
        output_path: Path,
        override: FlagOverride | None = None,
    ) -> FlagAttribution:
        resolved_title = self._resolve_commons_title(country=country, override=override)
        info = self._fetch_commons_image_info(
            title=resolved_title,
            thumb_height=self.cfg.output.height_px,
        )
        image_bytes = self._download_flag_image(info)
        self._write_normalized_png(
            image_bytes=image_bytes,
            output_path=output_path,
        )
        return self._build_attribution(country.iso3, info)

    def _resolve_commons_title(
        self,
        *,
        country: CountrySpec,
        override: FlagOverride | None,
    ) -> str:
        if override is not None:
            return _normalize_file_title(override.commons_file_title)

        mapping = self._load_iso3_to_title()
        direct = mapping.get(country.iso3)
        if direct is not None:
            return direct

        for candidate in _fallback_title_candidates(country):
            canonical = self._probe_title(candidate)
            if canonical is not None:
                return canonical

        raise RuntimeError(
            f"Could not resolve Wikimedia flag title for {country.iso3} ({country.name_en})"
        )

    def _load_iso3_to_title(self) -> dict[str, str]:
        if self._iso3_to_title is not None:
            return self._iso3_to_title

        query = """
SELECT ?iso3 ?flag WHERE {
  ?item wdt:P298 ?iso3 ;
        wdt:P41 ?flag .
}
""".strip()
        try:
            response = self._request_get(
                _WIKIDATA_SPARQL_URL,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            payload = response.json()
        except Exception as exc:  # pragma: no cover - network variability
            _LOGGER.warning("Failed loading Wikidata ISO3->flag mapping: %s", exc)
            self._iso3_to_title = {}
            return self._iso3_to_title

        buckets: dict[str, list[str]] = {}
        bindings = payload.get("results", {}).get("bindings", [])
        if isinstance(bindings, list):
            for row in bindings:
                if not isinstance(row, dict):
                    continue
                iso_raw = row.get("iso3", {}).get("value")
                flag_raw = row.get("flag", {}).get("value")
                if not isinstance(iso_raw, str) or not isinstance(flag_raw, str):
                    continue
                iso3 = iso_raw.strip().upper()
                if len(iso3) != 3:
                    continue
                title = _title_from_special_filepath(flag_raw)
                if title is None:
                    continue
                buckets.setdefault(iso3, []).append(title)

        resolved: dict[str, str] = {}
        for iso3, candidates in buckets.items():
            if not candidates:
                continue
            resolved[iso3] = _choose_preferred_title(candidates)
        self._iso3_to_title = resolved
        return resolved

    def _fetch_commons_image_info(self, *, title: str, thumb_height: int) -> _CommonsImageInfo:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlheight": max(int(thumb_height), 1),
        }
        response = self._request_get(
            _COMMONS_API_URL,
            params=params,
        )
        payload = response.json()
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list) or not pages:
            raise RuntimeError(f"Commons returned no pages for title '{title}'")
        page = pages[0]
        if not isinstance(page, dict):
            raise RuntimeError(f"Commons returned invalid page payload for '{title}'")
        if bool(page.get("missing")):
            raise RuntimeError(f"Commons file not found: '{title}'")

        canonical_title = page.get("title")
        if not isinstance(canonical_title, str) or not canonical_title.strip():
            canonical_title = title

        imageinfo_list = page.get("imageinfo", [])
        if not isinstance(imageinfo_list, list) or not imageinfo_list:
            raise RuntimeError(f"No imageinfo returned for '{canonical_title}'")
        imageinfo = imageinfo_list[0]
        if not isinstance(imageinfo, dict):
            raise RuntimeError(f"Invalid imageinfo payload for '{canonical_title}'")

        source_url = imageinfo.get("url")
        if not isinstance(source_url, str) or not source_url.strip():
            raise RuntimeError(f"Missing source URL for '{canonical_title}'")
        thumb_url_raw = imageinfo.get("thumburl")
        thumb_url = thumb_url_raw.strip() if isinstance(thumb_url_raw, str) and thumb_url_raw.strip() else None
        mime_raw = imageinfo.get("mime")
        mime = mime_raw.strip() if isinstance(mime_raw, str) and mime_raw.strip() else None

        extmetadata_raw = imageinfo.get("extmetadata")
        extmetadata: Mapping[str, Any]
        if isinstance(extmetadata_raw, Mapping):
            extmetadata = extmetadata_raw
        else:
            extmetadata = {}

        return _CommonsImageInfo(
            canonical_title=canonical_title,
            source_url=source_url,
            thumb_url=thumb_url,
            mime=mime,
            extmetadata=extmetadata,
        )

    def _download_flag_image(self, info: _CommonsImageInfo) -> bytes:
        canonical_lower = info.canonical_title.casefold()
        is_svg = (info.mime is not None and "svg" in info.mime.casefold()) or canonical_lower.endswith(".svg")

        if is_svg and self.cfg.svg.prefer_svg:
            try:
                svg_bytes = self._download_bytes(info.source_url)
            except Exception as exc:
                _LOGGER.warning(
                    "SVG download failed for %s; falling back to PNG thumbnail: %s",
                    info.canonical_title,
                    exc,
                )
            else:
                converted = self._try_convert_svg(svg_bytes)
                if converted is not None:
                    return converted

        download_url = info.thumb_url or info.source_url
        if download_url.casefold().endswith(".svg"):
            raise RuntimeError(
                f"No usable PNG thumbnail URL available for '{info.canonical_title}'"
            )
        return self._download_bytes(download_url)

    def _try_convert_svg(self, svg_bytes: bytes) -> bytes | None:
        try:
            import cairosvg  # type: ignore[import-untyped]
        except Exception:
            return None
        try:
            return cairosvg.svg2png(
                bytestring=svg_bytes,
                output_height=max(int(self.cfg.output.height_px), 1),
            )
        except Exception as exc:
            _LOGGER.warning("SVG conversion failed; falling back to Wikimedia PNG thumbnail: %s", exc)
            return None

    def _download_bytes(self, url: str) -> bytes:
        response = self._request_get(url)
        return response.content

    def _write_normalized_png(
        self,
        *,
        image_bytes: bytes,
        output_path: Path,
    ) -> None:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgba = image.convert("RGBA")

        target_height = max(int(self.cfg.output.height_px), 1)
        scale = target_height / max(rgba.height, 1)
        target_width = max(int(round(rgba.width * scale)), 1)

        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        resized = rgba.resize((target_width, target_height), resample=resampling)

        padding = max(int(self.cfg.output.padding_px), 0)
        canvas_width = target_width + (padding * 2)
        canvas_height = target_height + (padding * 2)
        background = _resolve_background_rgba(self.cfg.output.background)
        canvas = Image.new("RGBA", (canvas_width, canvas_height), background)
        canvas.paste(resized, (padding, padding), resized)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.output.background.casefold() == "white":
            canvas.convert("RGB").save(output_path, format="PNG")
        else:
            canvas.save(output_path, format="PNG")

    def _build_attribution(self, iso3: str, info: _CommonsImageInfo) -> FlagAttribution:
        license_name = (
            _clean_extmetadata_value(info.extmetadata, "LicenseShortName")
            or _clean_extmetadata_value(info.extmetadata, "License")
            or "Unknown"
        )
        author = _clean_extmetadata_value(info.extmetadata, "Artist")
        return FlagAttribution(
            iso3=iso3,
            commons_title=info.canonical_title,
            source_url=info.source_url,
            license_name=license_name,
            author=author,
            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def _probe_title(self, title: str) -> str | None:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": title,
        }
        response = self._request_get(
            _COMMONS_API_URL,
            params=params,
        )
        payload = response.json()
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list) or not pages:
            return None
        page = pages[0]
        if not isinstance(page, dict) or bool(page.get("missing")):
            return None
        page_title = page.get("title")
        if not isinstance(page_title, str) or not page_title.strip():
            return None
        return page_title.strip()

    def _request_get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> requests.Response:
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            self._wait_for_request_slot()
            response = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.cfg.request_timeout_s,
            )
            if response.status_code not in _RETRYABLE_HTTP_STATUS:
                response.raise_for_status()
                return response
            if attempt >= self._max_retries:
                response.raise_for_status()
            delay_s = self._compute_retry_delay_s(response=response, attempt=attempt)
            _LOGGER.warning(
                "Retryable response %s for %s; retrying in %.1fs (%d/%d)",
                response.status_code,
                response.url,
                delay_s,
                attempt + 1,
                self._max_retries,
            )
            response.close()
            time.sleep(delay_s)
        raise RuntimeError("Unreachable retry loop in Wikimedia flag fetcher")

    def _wait_for_request_slot(self) -> None:
        if self._min_request_interval_s <= 0:
            self._last_request_started_at = time.monotonic()
            return
        now = time.monotonic()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            if elapsed < self._min_request_interval_s:
                time.sleep(self._min_request_interval_s - elapsed)
        self._last_request_started_at = time.monotonic()

    def _compute_retry_delay_s(self, *, response: requests.Response, attempt: int) -> float:
        retry_after_s = _parse_retry_after_seconds(response.headers.get("Retry-After"))
        exponential_s = self._retry_backoff_s * (2**attempt)
        chosen = max(exponential_s, retry_after_s)
        return min(chosen, 300.0)


def _load_existing_attributions(path: Path | None) -> dict[str, FlagAttribution]:
    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}
    out: dict[str, FlagAttribution] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        parsed = FlagAttribution.from_dict(item)
        if parsed is None:
            continue
        out[parsed.iso3] = parsed
    return out


def _normalize_file_title(title: str) -> str:
    raw = title.strip().replace("_", " ")
    if raw.casefold().startswith("file:"):
        body = raw[5:].strip()
    else:
        body = raw
    if not body:
        raise ValueError("commons file title is empty")
    return f"File:{body}"


def _title_from_special_filepath(url: str) -> str | None:
    marker = "/Special:FilePath/"
    if marker not in url:
        return None
    name = url.split(marker, 1)[1]
    name = name.split("?", 1)[0]
    name = unquote(name).strip()
    if not name:
        return None
    return _normalize_file_title(f"File:{name}")


def _choose_preferred_title(candidates: Sequence[str]) -> str:
    ranked = sorted(candidates, key=_title_rank_key)
    return ranked[0]


def _title_rank_key(title: str) -> tuple[int, int, str]:
    lower = title.casefold()
    extension = lower.rsplit(".", 1)[-1] if "." in lower else ""
    extension_rank = {"svg": 0, "png": 1}.get(extension, 9)
    return (extension_rank, len(title), lower)


def _fallback_title_candidates(country: CountrySpec) -> Iterable[str]:
    names: list[str] = [country.name_en, country.ne_admin_name, *country.aliases]
    seen: set[str] = set()
    for name in names:
        stripped = name.strip()
        if not stripped:
            continue
        for extension in ("svg", "png"):
            candidate = _normalize_file_title(f"File:Flag of {stripped}.{extension}")
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _clean_extmetadata_value(metadata: Mapping[str, Any], key: str) -> str | None:
    entry = metadata.get(key)
    if not isinstance(entry, Mapping):
        return None
    raw = entry.get("value")
    if not isinstance(raw, str):
        return None
    cleaned = html.unescape(raw)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _parse_retry_after_seconds(raw: str | None) -> float:
    if raw is None:
        return 0.0
    value = raw.strip()
    if not value:
        return 0.0
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    return max(parsed, 0.0)


def _resolve_background_rgba(background: str) -> tuple[int, int, int, int]:
    if background.casefold() == "white":
        return (255, 255, 255, 255)
    return (0, 0, 0, 0)


def _format_code_list(values: list[str], limit: int = 12) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    shown = ", ".join(values[:limit])
    return f"{shown}, ... (+{len(values) - limit} more)"
