"""Validation layer for config and input datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import AppConfig
from .cities import CityOverride, load_cities_overrides, names_match
from .countries import load_un_members
from .flags import load_flags_overrides
from .io_ne import NaturalEarthRepository
from .models import CountrySpec
from .qa import write_qa_index
from .render_overrides import load_render_overrides


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
        self._validate_render_overrides(report)
        self._validate_natural_earth(
            report,
            countries=countries,
            city_overrides=city_overrides,
            strict_data_files=strict_data_files,
        )
        self._validate_generated_artifacts(report, countries=countries)
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

    def _validate_render_overrides(self, report: ValidationReport) -> None:
        try:
            render_overrides = load_render_overrides(self.cfg.paths.render_overrides)
            report.add_info(f"Loaded {len(render_overrides)} render override entries")
        except Exception as exc:
            report.add_error(f"Failed parsing render overrides: {exc}")

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
        missing_named_capitals: list[str] = []
        auto_relabel_capitals: list[str] = []
        override_capital_used = 0
        manual_capital_used = 0

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
            expected_capital_found = any(
                names_match(city.name, country.capital) for city in city_candidates
            )

            override = city_overrides.get(country.iso3)
            override_capital_found = False
            has_manual_capital = False
            if override is not None:
                if override.capital is not None:
                    override_capital_found = any(
                        names_match(city.name, override.capital) for city in city_candidates
                    )
                if override.manual_capital is not None:
                    has_manual_capital = True

            if (
                not expected_capital_found
                and not override_capital_found
                and not has_manual_capital
                and not city_candidates
            ):
                missing_capitals.append(country.iso3)
                missing_named_capitals.append(f"{country.iso3}({country.capital})")
            elif (
                not expected_capital_found
                and not override_capital_found
                and not has_manual_capital
                and city_candidates
            ):
                auto_relabel_capitals.append(f"{country.iso3}({country.capital})")
            if override_capital_found and not expected_capital_found:
                override_capital_used += 1
            if has_manual_capital and not expected_capital_found:
                manual_capital_used += 1

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
                "UN countries whose configured capitals are not present in populated places and no override is available: "
                f"{_format_code_list(sorted(missing_named_capitals))}",
                strict_data_files=strict_data_files,
            )
            for iso3 in sorted(missing_capitals):
                report.add_info(
                    f"Capital override hint for {iso3}: no populated-place candidates found in dataset."
                )
        if auto_relabel_capitals:
            report.add_warning(
                "Configured capital names not found in populated places; selection will auto-relabel suspected "
                f"capital cities: {_format_code_list(sorted(auto_relabel_capitals))}"
            )

        report.add_info(
            "Natural Earth check summary: "
            f"countries={len(countries)}, "
            f"admin0_rows={len(admin0_df)}, populated_places_rows={len(places_df)}, "
            f"capital_name_overrides_used={override_capital_used}, "
            f"manual_capital_overrides_used={manual_capital_used}, "
            f"auto_capital_relabels_planned={len(auto_relabel_capitals)}"
        )

    def _validate_generated_artifacts(
        self,
        report: ValidationReport,
        *,
        countries: list[CountrySpec],
    ) -> None:
        if not countries:
            return

        missing_maps: list[str] = []
        missing_flags: list[str] = []
        for country in countries:
            map_path = self.cfg.paths.maps_dir / f"map_{country.iso3}.png"
            flag_path = self.cfg.paths.flags_dir / f"flag_{country.iso3}.png"
            if not map_path.exists():
                missing_maps.append(country.iso3)
            if not flag_path.exists():
                missing_flags.append(country.iso3)

        map_present = len(countries) - len(missing_maps)
        flag_present = len(countries) - len(missing_flags)
        report.add_info(
            "Media coverage summary: "
            f"maps_present={map_present}/{len(countries)}, "
            f"flags_present={flag_present}/{len(countries)}"
        )

        if missing_maps:
            report.add_warning(
                "Missing rendered maps: "
                f"{_format_code_list(sorted(missing_maps))}"
            )
        if missing_flags:
            report.add_warning(
                "Missing flags (expected before Milestone 4): "
                f"{_format_code_list(sorted(missing_flags))}"
            )

        if self.cfg.qa.generate_index:
            try:
                qa_path = write_qa_index(
                    countries=countries,
                    maps_dir=self.cfg.paths.maps_dir,
                    flags_dir=self.cfg.paths.flags_dir,
                    output_html=self.cfg.paths.qa_dir / "index.html",
                    thumbnail_width_px=self.cfg.qa.thumbnail_width_px,
                    max_columns=self.cfg.qa.max_columns,
                )
            except Exception as exc:
                report.add_warning(f"Failed generating QA index: {exc}")
            else:
                report.add_info(f"QA index updated at {qa_path}")

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
