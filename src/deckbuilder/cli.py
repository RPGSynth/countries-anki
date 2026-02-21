"""CLI entrypoint for countries-anki deckbuilder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .city_selection import format_city_selection_lines, run_city_selection
from .config import AppConfig, load_config
from .countries import load_un_members
from .flags import format_flag_lines, run_fetch_flags
from .inspect_report import generate_inspection_report
from .models import BuildManifest, CountrySpec
from .qa import write_qa_index
from .render import format_render_lines, run_render_maps
from .util import detect_git_commit, ensure_directories, sha256_file, setup_logging, write_json
from .validate import Validator, format_report_lines

LOGGER = logging.getLogger("deckbuilder.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deckbuilder",
        description="Countries Anki deck builder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="config.yaml", help="Path to YAML config.")
        p.add_argument("--verbose", action="store_true", help="Enable debug logs.")

    build_p = subparsers.add_parser("build", help="Run full build pipeline.")
    add_common(build_p)
    build_p.add_argument(
        "--strict-data-files",
        action="store_true",
        help="Fail if Natural Earth dataset files are missing.",
    )

    validate_p = subparsers.add_parser("validate", help="Validate config and input files.")
    add_common(validate_p)
    validate_p.add_argument(
        "--strict-data-files",
        action="store_true",
        help="Treat missing Natural Earth files as validation errors.",
    )

    select_p = subparsers.add_parser(
        "select-cities",
        help="Run deterministic city selection and write audit JSON.",
    )
    add_common(select_p)
    select_p.add_argument(
        "--strict-data-files",
        action="store_true",
        help="Treat missing Natural Earth files as hard errors.",
    )

    render_p = subparsers.add_parser("render-maps", help="Render maps only.")
    add_common(render_p)
    render_p.add_argument(
        "--strict-data-files",
        action="store_true",
        help="Treat missing Natural Earth files as hard errors.",
    )
    render_p.add_argument(
        "--country",
        action="append",
        default=[],
        help="ISO3 filter for rendering. Can be repeated.",
    )
    render_p.add_argument(
        "--limit-countries",
        type=int,
        default=None,
        help="Render only first N ISO3-sorted countries.",
    )
    render_p.add_argument(
        "--debug-render",
        action="store_true",
        help="Enable live per-country render debug logs.",
    )
    render_p.add_argument(
        "--clean-maps",
        action="store_true",
        help="Delete existing map_*.png files before rendering.",
    )
    render_p.add_argument(
        "--force-render",
        action="store_true",
        help="Re-render maps even when map_*.png already exists.",
    )

    flags_p = subparsers.add_parser("fetch-flags", help="Fetch flags only.")
    add_common(flags_p)
    flags_p.add_argument(
        "--country",
        action="append",
        default=[],
        help="ISO3 filter for flag fetching. Can be repeated.",
    )
    flags_p.add_argument(
        "--limit-countries",
        type=int,
        default=None,
        help="Fetch only first N ISO3-sorted countries.",
    )
    flags_p.add_argument(
        "--clean-flags",
        action="store_true",
        help="Delete existing flag_*.png files before fetching.",
    )

    deck_p = subparsers.add_parser("build-deck", help="Build Anki deck only (not implemented yet).")
    add_common(deck_p)

    inspect_p = subparsers.add_parser(
        "inspect",
        help="Generate visual + JSON inspection report for data joins and city selection.",
    )
    add_common(inspect_p)
    inspect_p.add_argument(
        "--country",
        action="append",
        default=[],
        help="ISO3 filter. Can be repeated. If omitted, first N countries are shown.",
    )
    inspect_p.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Max countries in report when --country is not provided.",
    )

    return parser


def _load_and_setup(args: argparse.Namespace) -> AppConfig:
    cfg = load_config(args.config)
    log_path = cfg.paths.logs_dir / "build.log"
    setup_logging(log_path, verbose=args.verbose)
    ensure_directories(cfg.paths.build_directories)
    return cfg


def _run_validate(cfg: AppConfig, *, strict_data_files: bool) -> int:
    report = Validator(cfg).run(strict_data_files=strict_data_files)
    for line in format_report_lines(report):
        LOGGER.info(line)
    return 0 if report.ok else 1


def _run_build(cfg: AppConfig, *, strict_data_files: bool) -> int:
    LOGGER.info("Starting build pipeline.")

    report = Validator(cfg).run(strict_data_files=strict_data_files)
    for line in format_report_lines(report):
        LOGGER.info(line)
    if not report.ok:
        LOGGER.error("Build aborted due to validation errors.")
        return 1

    city_selection_report = run_city_selection(cfg, strict_data_files=strict_data_files)
    for line in format_city_selection_lines(city_selection_report):
        LOGGER.info(line)
    if not city_selection_report.ok:
        LOGGER.error("Build aborted due to city selection errors.")
        return 1

    render_report = run_render_maps(
        cfg,
        strict_data_files=strict_data_files,
        selection_manifest_path=city_selection_report.output_path,
    )
    for line in format_render_lines(render_report):
        LOGGER.info(line)
    if not render_report.ok:
        LOGGER.error("Build aborted due to map rendering errors.")
        return 1

    flag_report = run_fetch_flags(cfg)
    for line in format_flag_lines(flag_report):
        LOGGER.info(line)
    if not flag_report.ok:
        LOGGER.error("Build aborted due to flag fetch errors.")
        return 1

    countries = load_un_members(cfg.paths.un_members)
    qa_index_path: Path | None = None
    if cfg.qa.generate_index:
        qa_index_path = write_qa_index(
            countries=countries,
            maps_dir=cfg.paths.maps_dir,
            flags_dir=cfg.paths.flags_dir,
            output_html=cfg.paths.qa_dir / "index.html",
            thumbnail_width_px=cfg.qa.thumbnail_width_px,
            max_columns=cfg.qa.max_columns,
        )
        LOGGER.info("QA index generated at %s", qa_index_path)

    LOGGER.info("Deck step: stub (Milestone 5 pending).")

    if cfg.build.write_manifest:
        manifest = BuildManifest.create(
            config_hash_sha256=sha256_file(cfg.source_path),
            git_commit=detect_git_commit(cfg.source_path.parent),
            steps={
                "validate": "ok",
                "select_cities": "ok" if city_selection_report.output_path else "skipped",
                "render_maps": "ok" if render_report.ok else "error",
                "fetch_flags": "ok" if flag_report.ok else "error",
                "build_deck": "stub",
                "qa_index": "ok" if qa_index_path else "skipped",
            },
            artifacts={
                "qa_index": str(qa_index_path) if qa_index_path else "",
                "city_selection": (
                    str(city_selection_report.output_path)
                    if city_selection_report.output_path is not None
                    else ""
                ),
                "maps_dir": str(cfg.paths.maps_dir),
                "flags_dir": str(cfg.paths.flags_dir),
                "flags_attribution": str(cfg.paths.attribution_dir / "flags.json"),
                "output_apkg": str(cfg.project.output_apkg),
            },
        )
        manifest_path = cfg.paths.manifests_dir / "build_manifest.json"
        write_json(manifest_path, manifest.to_dict())
        LOGGER.info("Build manifest written to %s", manifest_path)

    LOGGER.info("Build finished.")
    return 0


def _stub_command(name: str) -> int:
    LOGGER.error("%s is intentionally not implemented yet in this skeleton.", name)
    return 2


def _run_inspect(cfg: AppConfig, *, countries: Sequence[str], limit: int) -> int:
    try:
        html_path, json_path = generate_inspection_report(
            cfg,
            country_filters=countries,
            limit=limit,
        )
    except Exception as exc:
        LOGGER.error("Inspection report failed: %s", exc)
        return 1
    LOGGER.info("Inspection HTML report written to %s", html_path)
    LOGGER.info("Inspection JSON report written to %s", json_path)
    return 0


def _run_select_cities(cfg: AppConfig, *, strict_data_files: bool) -> int:
    report = run_city_selection(cfg, strict_data_files=strict_data_files)
    for line in format_city_selection_lines(report):
        LOGGER.info(line)
    return 0 if report.ok else 1


def _run_render_maps(
    cfg: AppConfig,
    *,
    strict_data_files: bool,
    countries: Sequence[str],
    limit_countries: int | None,
    debug_render: bool,
    clean_maps: bool,
    force_render: bool,
) -> int:
    effective_limit = limit_countries
    if debug_render and effective_limit is None and not countries:
        effective_limit = 3
        LOGGER.info(
            "[render-debug] No country filter/limit provided; using automatic limit=3 for fast debug."
        )

    if clean_maps:
        removed = 0
        for path in cfg.paths.maps_dir.glob("map_*.png"):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                LOGGER.warning("Failed removing old map file %s: %s", path, exc)
        LOGGER.info("Cleaned %d existing map PNG files from %s", removed, cfg.paths.maps_dir)

    city_selection_report = run_city_selection(cfg, strict_data_files=strict_data_files)
    for line in format_city_selection_lines(city_selection_report):
        LOGGER.info(line)
    if not city_selection_report.ok:
        return 1

    render_report = run_render_maps(
        cfg,
        strict_data_files=strict_data_files,
        selection_manifest_path=city_selection_report.output_path,
        country_filter=countries,
        limit_countries=effective_limit,
        debug_render=debug_render,
        skip_existing=not force_render,
    )
    for line in format_render_lines(render_report):
        LOGGER.info(line)
    if not render_report.ok:
        return 1

    if cfg.qa.generate_index:
        qa_countries = _resolve_render_scope_countries(
            cfg,
            countries=countries,
            limit_countries=effective_limit,
        )
        if countries or effective_limit is not None:
            LOGGER.info(
                "QA index scoped to render selection (%d countries).",
                len(qa_countries),
            )
        else:
            qa_countries = load_un_members(cfg.paths.un_members)
        qa_index_path = write_qa_index(
            countries=qa_countries,
            maps_dir=cfg.paths.maps_dir,
            flags_dir=cfg.paths.flags_dir,
            output_html=cfg.paths.qa_dir / "index.html",
            thumbnail_width_px=cfg.qa.thumbnail_width_px,
            max_columns=cfg.qa.max_columns,
            flag_status_mode="render_only",
        )
        LOGGER.info("QA index generated at %s", qa_index_path)
    return 0


def _run_fetch_flags(
    cfg: AppConfig,
    *,
    countries: Sequence[str],
    limit_countries: int | None,
    clean_flags: bool,
) -> int:
    report = run_fetch_flags(
        cfg,
        country_filter=countries,
        limit_countries=limit_countries,
        clean_flags=clean_flags,
    )
    for line in format_flag_lines(report):
        LOGGER.info(line)
    if not report.ok:
        return 1

    if cfg.qa.generate_index:
        qa_countries = _resolve_render_scope_countries(
            cfg,
            countries=countries,
            limit_countries=limit_countries,
        )
        if countries or limit_countries is not None:
            LOGGER.info(
                "QA index scoped to flag selection (%d countries).",
                len(qa_countries),
            )
        else:
            qa_countries = load_un_members(cfg.paths.un_members)
        qa_index_path = write_qa_index(
            countries=qa_countries,
            maps_dir=cfg.paths.maps_dir,
            flags_dir=cfg.paths.flags_dir,
            output_html=cfg.paths.qa_dir / "index.html",
            thumbnail_width_px=cfg.qa.thumbnail_width_px,
            max_columns=cfg.qa.max_columns,
            flag_status_mode="strict",
        )
        LOGGER.info("QA index generated at %s", qa_index_path)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    cfg = _load_and_setup(args)
    command = str(args.command)
    if command == "build":
        strict = bool(args.strict_data_files) or cfg.country_policy.strict
        return _run_build(cfg, strict_data_files=strict)
    if command == "validate":
        strict = bool(args.strict_data_files) or cfg.country_policy.strict
        return _run_validate(cfg, strict_data_files=strict)
    if command == "select-cities":
        strict = bool(args.strict_data_files) or cfg.country_policy.strict
        return _run_select_cities(cfg, strict_data_files=strict)
    if command == "render-maps":
        strict = bool(args.strict_data_files) or cfg.country_policy.strict
        countries = [str(item) for item in args.country]
        return _run_render_maps(
            cfg,
            strict_data_files=strict,
            countries=countries,
            limit_countries=args.limit_countries,
            debug_render=bool(args.debug_render),
            clean_maps=bool(args.clean_maps),
            force_render=bool(args.force_render),
        )
    if command == "fetch-flags":
        countries = [str(item) for item in args.country]
        return _run_fetch_flags(
            cfg,
            countries=countries,
            limit_countries=args.limit_countries,
            clean_flags=bool(args.clean_flags),
        )
    if command == "build-deck":
        return _stub_command("build-deck")
    if command == "inspect":
        countries = [str(item) for item in args.country]
        return _run_inspect(cfg, countries=countries, limit=int(args.limit))
    raise ValueError(f"Unknown command: {command}")


def _resolve_render_scope_countries(
    cfg: AppConfig,
    *,
    countries: Sequence[str],
    limit_countries: int | None,
) -> list[CountrySpec]:
    all_countries = sorted(load_un_members(cfg.paths.un_members), key=lambda item: item.iso3)
    requested = {item.strip().upper() for item in countries if item and item.strip()}
    if requested:
        all_countries = [country for country in all_countries if country.iso3 in requested]
    if limit_countries is not None:
        all_countries = all_countries[:limit_countries]
    return all_countries


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
