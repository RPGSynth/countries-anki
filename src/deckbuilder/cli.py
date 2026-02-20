"""CLI entrypoint for countries-anki deckbuilder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .city_selection import format_city_selection_lines, run_city_selection
from .config import AppConfig, load_config
from .countries import load_un_members
from .inspect_report import generate_inspection_report
from .models import BuildManifest
from .qa import write_qa_index
from .util import detect_git_commit, ensure_directories, sha256_file, setup_logging, write_json
from .validate import Validator, format_report_lines

LOGGER = logging.getLogger("deckbuilder.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deckbuilder",
        description="Countries Anki deck builder (skeleton architecture).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="config.yaml", help="Path to YAML config.")
        p.add_argument("--verbose", action="store_true", help="Enable debug logs.")

    build_p = subparsers.add_parser("build", help="Run full build pipeline (skeleton mode).")
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

    render_p = subparsers.add_parser("render-maps", help="Render maps only (not implemented yet).")
    add_common(render_p)

    flags_p = subparsers.add_parser("fetch-flags", help="Fetch flags only (not implemented yet).")
    add_common(flags_p)

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
    LOGGER.info("Starting build pipeline in skeleton mode.")

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

    LOGGER.info("Render step: stub (Milestone 3 pending).")
    LOGGER.info("Flag step: stub (Milestone 4 pending).")
    LOGGER.info("Deck step: stub (Milestone 5 pending).")

    if cfg.build.write_manifest:
        manifest = BuildManifest.create(
            config_hash_sha256=sha256_file(cfg.source_path),
            git_commit=detect_git_commit(cfg.source_path.parent),
            steps={
                "validate": "ok",
                "select_cities": "ok" if city_selection_report.output_path else "skipped",
                "render_maps": "stub",
                "fetch_flags": "stub",
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
                "output_apkg": str(cfg.project.output_apkg),
            },
        )
        manifest_path = cfg.paths.manifests_dir / "build_manifest.json"
        write_json(manifest_path, manifest.to_dict())
        LOGGER.info("Build manifest written to %s", manifest_path)

    LOGGER.info("Build finished (skeleton mode).")
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
        return _stub_command("render-maps")
    if command == "fetch-flags":
        return _stub_command("fetch-flags")
    if command == "build-deck":
        return _stub_command("build-deck")
    if command == "inspect":
        countries = [str(item) for item in args.country]
        return _run_inspect(cfg, countries=countries, limit=int(args.limit))
    raise ValueError(f"Unknown command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
