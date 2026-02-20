"""Validation layer for config and input datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import AppConfig
from .countries import load_un_members
from .flags import load_flags_overrides
from .cities import load_cities_overrides


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.infos.append(msg)


class Validator:
    """Top-level input and schema validator."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    def run(self, *, strict_data_files: bool) -> ValidationReport:
        report = ValidationReport()
        self._validate_config_paths(report, strict_data_files=strict_data_files)
        self._validate_un_members(report)
        self._validate_overrides(report)
        return report

    def _validate_config_paths(self, report: ValidationReport, *, strict_data_files: bool) -> None:
        for path in self.cfg.paths.required_input_files:
            if not path.exists():
                report.add_error(f"Missing required input file: {path}")

        if strict_data_files:
            self._check_exists(report, self.cfg.paths.ne_admin0_countries, as_error=True)
            self._check_exists(report, self.cfg.paths.ne_populated_places, as_error=True)
        else:
            self._check_exists(report, self.cfg.paths.ne_admin0_countries, as_error=False)
            self._check_exists(report, self.cfg.paths.ne_populated_places, as_error=False)

    def _validate_un_members(self, report: ValidationReport) -> None:
        path = self.cfg.paths.un_members
        if not path.exists():
            return
        try:
            countries = load_un_members(path)
        except Exception as exc:
            report.add_error(f"Failed parsing UN members file '{path}': {exc}")
            return
        if not countries:
            report.add_error(f"UN members file is empty: {path}")
            return
        report.add_info(f"Loaded {len(countries)} UN member records from {path}")

        expected_count = 193
        if len(countries) != expected_count:
            report.add_warning(
                f"Expected {expected_count} UN members but found {len(countries)}. "
                "Verify authoritative source."
            )

    def _validate_overrides(self, report: ValidationReport) -> None:
        try:
            city_overrides = load_cities_overrides(self.cfg.paths.cities_overrides)
            report.add_info(f"Loaded {len(city_overrides)} city override entries")
        except Exception as exc:
            report.add_error(f"Failed parsing cities overrides: {exc}")

        try:
            flag_overrides = load_flags_overrides(self.cfg.paths.flags_overrides)
            report.add_info(f"Loaded {len(flag_overrides)} flag override entries")
        except Exception as exc:
            report.add_error(f"Failed parsing flags overrides: {exc}")

    @staticmethod
    def _check_exists(report: ValidationReport, path: Path, *, as_error: bool) -> None:
        if path.exists():
            return
        msg = f"Missing dataset file: {path}"
        if as_error:
            report.add_error(msg)
        else:
            report.add_warning(msg)


def format_report_lines(report: ValidationReport) -> Iterable[str]:
    if report.infos:
        for info in report.infos:
            yield f"[INFO] {info}"
    if report.warnings:
        for warning in report.warnings:
            yield f"[WARN] {warning}"
    if report.errors:
        for error in report.errors:
            yield f"[ERROR] {error}"
    if report.ok:
        yield "[OK] Validation completed with no errors."
