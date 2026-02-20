"""Validation layer for config and input datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import AppConfig
from .cities import CityOverride, load_cities_overrides
from .countries import load_un_members
from .flags import load_flags_overrides
from .io_ne import NaturalEarthRepository
from .models import CountrySpec


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
        countries = self._validate_un_members(report)
        city_overrides = self._validate_cities_overrides(report)
        self._validate_flags_overrides(report)
        self._validate_natural_earth(
            report,
            countries=countries,
            city_overrides=city_overrides,
            strict_data_files=strict_data_files,
        )
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

    def _validate_un_members(self, report: ValidationReport) -> list[CountrySpec]:
        path = self.cfg.paths.un_members
        if not path.exists():
            return []
        try:
            countries = load_un_members(path)
        except Exception as exc:
            report.add_error(f"Failed parsing UN members file '{path}': {exc}")
            return []
        if not countries:
            report.add_error(f"UN members file is empty: {path}")
            return []
        report.add_info(f"Loaded {len(countries)} UN member records from {path}")

        expected_count = 193
        if len(countries) != expected_count:
            report.add_warning(
                f"Expected {expected_count} UN members but found {len(countries)}. "
                "Verify authoritative source."
            )
        return countries

    def _validate_cities_overrides(self, report: ValidationReport) -> dict[str, CityOverride]:
        try:
            city_overrides = load_cities_overrides(self.cfg.paths.cities_overrides)
            report.add_info(f"Loaded {len(city_overrides)} city override entries")
        except Exception as exc:
            report.add_error(f"Failed parsing cities overrides: {exc}")
            return {}
        return city_overrides

    def _validate_flags_overrides(self, report: ValidationReport) -> None:
        try:
            flag_overrides = load_flags_overrides(self.cfg.paths.flags_overrides)
            report.add_info(f"Loaded {len(flag_overrides)} flag override entries")
        except Exception as exc:
            report.add_error(f"Failed parsing flags overrides: {exc}")

    def _validate_natural_earth(
        self,
        report: ValidationReport,
        *,
        countries: list[CountrySpec],
        city_overrides: dict[str, CityOverride],
        strict_data_files: bool,
    ) -> None:
        if not countries:
            return

        admin0_path = self.cfg.paths.ne_admin0_countries
        places_path = self.cfg.paths.ne_populated_places
        if not admin0_path.exists() or not places_path.exists():
            report.add_info("Skipping Natural Earth join/capital checks because dataset files are missing.")
            return

        repo = NaturalEarthRepository(admin0_path, places_path)
        try:
            admin0_df = repo.load_admin0()
            places_df = repo.load_populated_places()
        except Exception as exc:
            self._add_quality_issue(
                report,
                f"Failed loading Natural Earth datasets: {exc}",
                strict_data_files=strict_data_files,
            )
            return

        country_iso = {country.iso3 for country in countries}

        try:
            admin_iso_col = repo.detect_admin0_iso_column(admin0_df, iso_allowlist=country_iso)
            places_iso_col = repo.detect_places_iso_column(places_df, iso_allowlist=country_iso)
        except Exception as exc:
            self._add_quality_issue(
                report,
                f"Natural Earth schema validation failed: {exc}",
                strict_data_files=strict_data_files,
            )
            return
        report.add_info(
            f"Natural Earth ISO columns selected: admin0={admin_iso_col}, populated_places={places_iso_col}"
        )

        admin_iso_values = {
            str(value).strip().upper()
            for value in admin0_df[admin_iso_col].dropna().tolist()
            if len(str(value).strip()) == 3
        }
        missing_in_admin0 = sorted(country_iso - admin_iso_values)
        missing_in_admin0_set = set(missing_in_admin0)
        if missing_in_admin0:
            self._add_quality_issue(
                report,
                "UN ISO3 missing from Natural Earth admin0: "
                f"{_format_code_list(missing_in_admin0)}",
                strict_data_files=strict_data_files,
            )

        invalid_geometry_count: list[str] = []
        multi_geometry_rows: list[str] = []
        missing_capitals: list[str] = []
        missing_capital_hints: dict[str, list[str]] = {}
        override_capital_used = 0

        for country in countries:
            geometry_rows = repo.extract_country_geometry(
                admin0_df,
                country.iso3,
                iso_col=admin_iso_col,
            )
            row_count = int(len(geometry_rows))
            if row_count == 0 and country.iso3 not in missing_in_admin0_set:
                invalid_geometry_count.append(country.iso3)
            elif row_count > 1:
                multi_geometry_rows.append(f"{country.iso3}({row_count})")

            if row_count > 0 and country.iso3 not in missing_in_admin0_set and "geometry" in geometry_rows:
                non_null_geometry = int(geometry_rows["geometry"].notna().sum())
                if non_null_geometry == 0:
                    invalid_geometry_count.append(country.iso3)

            city_candidates = repo.extract_city_candidates(
                places_df,
                iso3=country.iso3,
                capital_fields=self.cfg.cities.capital_fields,
                iso_col=places_iso_col,
            )
            has_capital_in_dataset = any(city.is_capital for city in city_candidates)
            has_capital_override = bool(
                country.iso3 in city_overrides
                and (
                    city_overrides[country.iso3].capital is not None
                    or city_overrides[country.iso3].manual_capital is not None
                )
            )
            if not has_capital_in_dataset and not has_capital_override:
                missing_capitals.append(country.iso3)
                if city_candidates:
                    ordered = sorted(
                        city_candidates,
                        key=lambda c: (
                            c.scalerank if c.scalerank is not None else 9999,
                            -(c.pop_max if c.pop_max is not None else -1),
                            c.name.casefold(),
                        ),
                    )
                    unique_names: list[str] = []
                    seen_names: set[str] = set()
                    for candidate in ordered:
                        folded = candidate.name.casefold()
                        if folded in seen_names:
                            continue
                        seen_names.add(folded)
                        unique_names.append(candidate.name)
                        if len(unique_names) >= 5:
                            break
                    missing_capital_hints[country.iso3] = unique_names
            if has_capital_override and not has_capital_in_dataset:
                override_capital_used += 1

        if invalid_geometry_count:
            self._add_quality_issue(
                report,
                "UN countries with missing/invalid geometry rows in admin0: "
                f"{_format_code_list(sorted(set(invalid_geometry_count)))}",
                strict_data_files=strict_data_files,
            )

        if multi_geometry_rows:
            report.add_warning(
                "UN countries with multiple admin0 rows; these should be merged/dissolved in rendering: "
                f"{_format_code_list(sorted(multi_geometry_rows))}"
            )

        if missing_capitals:
            self._add_quality_issue(
                report,
                "UN countries missing a detected capital in populated places and no override: "
                f"{_format_code_list(sorted(missing_capitals))}",
                strict_data_files=strict_data_files,
            )
            for iso3 in sorted(missing_capitals):
                candidates = missing_capital_hints.get(iso3, [])
                if candidates:
                    report.add_info(
                        f"Capital override hint for {iso3}: candidate city names = "
                        f"{_format_code_list(candidates, limit=5)}"
                    )
                else:
                    report.add_info(
                        f"Capital override hint for {iso3}: no populated-place candidates found in dataset."
                    )

        report.add_info(
            "Natural Earth check summary: "
            f"countries={len(countries)}, "
            f"admin0_rows={len(admin0_df)}, populated_places_rows={len(places_df)}, "
            f"capital_overrides_used={override_capital_used}"
        )

    @staticmethod
    def _check_exists(report: ValidationReport, path: Path, *, as_error: bool) -> None:
        if path.exists():
            return
        msg = f"Missing dataset file: {path}"
        if as_error:
            report.add_error(msg)
        else:
            report.add_warning(msg)

    def _add_quality_issue(
        self,
        report: ValidationReport,
        msg: str,
        *,
        strict_data_files: bool,
    ) -> None:
        hard_fail = strict_data_files or self.cfg.country_policy.strict
        if hard_fail:
            report.add_error(msg)
        else:
            report.add_warning(msg)


def _format_code_list(values: list[str], limit: int = 12) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    shown = ", ".join(values[:limit])
    return f"{shown}, ... (+{len(values) - limit} more)"


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
