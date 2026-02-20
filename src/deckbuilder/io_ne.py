"""Natural Earth dataset loading interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import CityRecord


def _first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    existing = {col.lower(): col for col in columns}
    for candidate in candidates:
        match = existing.get(candidate.lower())
        if match:
            return match
    return None


class NaturalEarthRepository:
    """Thin wrapper around Natural Earth file access.

    The methods are intentionally minimal in this skeleton. They define stable
    seams for the full implementation in subsequent milestones.
    """

    COUNTRY_ISO_COLUMNS = (
        "ADM0_A3",
        "ADM0_A3_US",
        "ADM0_A3_UN",
        "ISO_A3",
        "ISO_A3_EH",
        "SOV_A3",
        "WB_A3",
        "BRK_A3",
        "SU_A3",
        "GU_A3",
        "ISO3",
        "A3",
    )
    PLACE_ISO_COLUMNS = (
        "ADM0_A3",
        "ISO_A3",
        "SOV0_A3",
        "SOV_A3",
        "GU_A3",
        "WB_A3",
        "BRK_A3",
        "SU_A3",
        "ISO3",
        "A3",
    )
    PLACE_NAME_COLUMNS = ("NAME", "NAMEASCII", "name", "nameascii")
    PLACE_LON_COLUMNS = ("LONGITUDE", "LON", "longitude", "lon")
    PLACE_LAT_COLUMNS = ("LATITUDE", "LAT", "latitude", "lat")
    PLACE_SCALERANK_COLUMNS = ("SCALERANK", "scalerank")
    PLACE_POPMAX_COLUMNS = ("POP_MAX", "pop_max")

    def __init__(self, admin0_path: Path, populated_places_path: Path) -> None:
        self.admin0_path = admin0_path
        self.populated_places_path = populated_places_path

    def load_admin0(self) -> Any:
        """Load admin-0 country polygons via GeoPandas."""
        gpd = self._require_geopandas()
        return gpd.read_file(self.admin0_path)

    def load_populated_places(self) -> Any:
        """Load populated places via GeoPandas."""
        gpd = self._require_geopandas()
        return gpd.read_file(self.populated_places_path)

    def detect_admin0_iso_column(
        self,
        admin0_df: Any,
        *,
        iso_allowlist: set[str] | None = None,
    ) -> str:
        iso_col = _select_best_iso_column(
            admin0_df,
            self.COUNTRY_ISO_COLUMNS,
            iso_allowlist=iso_allowlist,
        )
        if iso_col is None:
            cols = ", ".join(str(c) for c in admin0_df.columns)
            raise ValueError(
                "Could not detect ISO3 column in Natural Earth admin0 data. "
                f"Available columns: {cols}"
            )
        return iso_col

    def detect_places_iso_column(
        self,
        places_df: Any,
        *,
        iso_allowlist: set[str] | None = None,
    ) -> str:
        iso_col = _select_best_iso_column(
            places_df,
            self.PLACE_ISO_COLUMNS,
            iso_allowlist=iso_allowlist,
        )
        if iso_col is None:
            cols = ", ".join(str(c) for c in places_df.columns)
            raise ValueError(
                "Could not detect country-ISO join column in populated places data. "
                f"Available columns: {cols}"
            )
        return iso_col

    def extract_country_geometry(self, admin0_df: Any, iso3: str, *, iso_col: str | None = None) -> Any:
        """Return geometry rows for one ISO3.

        Full polygon normalization (multipolygons, antimeridian handling)
        is deferred to rendering milestones.
        """
        col = iso_col or self.detect_admin0_iso_column(admin0_df)
        return admin0_df[admin0_df[col] == iso3]

    def extract_city_candidates(
        self,
        places_df: Any,
        *,
        iso3: str,
        capital_fields: Sequence[str] | None = None,
        iso_col: str | None = None,
    ) -> list[CityRecord]:
        """Extract city records for one country from populated places.

        This function already provides deterministic conversion into typed records.
        Additional heuristics can be layered later without changing callers.
        """
        place_iso_col = iso_col or self.detect_places_iso_column(places_df)

        name_col = _first_existing_column(places_df.columns, self.PLACE_NAME_COLUMNS)
        lon_col = _first_existing_column(places_df.columns, self.PLACE_LON_COLUMNS)
        lat_col = _first_existing_column(places_df.columns, self.PLACE_LAT_COLUMNS)
        scalerank_col = _first_existing_column(places_df.columns, self.PLACE_SCALERANK_COLUMNS)
        pop_col = _first_existing_column(places_df.columns, self.PLACE_POPMAX_COLUMNS)

        if name_col is None or lon_col is None or lat_col is None:
            raise ValueError("Populated places dataset missing required name/lon/lat columns")

        capital_cols: list[str] = []
        if capital_fields is not None:
            for field in capital_fields:
                col = _first_existing_column(places_df.columns, [field])
                if col:
                    capital_cols.append(col)

        subset = places_df[places_df[place_iso_col] == iso3]
        records: list[CityRecord] = []
        for row in subset.itertuples(index=False):
            row_dict = row._asdict()
            name_val = row_dict.get(name_col)
            lon_val = row_dict.get(lon_col)
            lat_val = row_dict.get(lat_col)
            if name_val is None or lon_val is None or lat_val is None:
                continue
            name = str(name_val).strip()
            if not name:
                continue
            scalerank = self._to_int_or_none(row_dict.get(scalerank_col)) if scalerank_col else None
            pop_max = self._to_int_or_none(row_dict.get(pop_col)) if pop_col else None
            is_capital = self._infer_capital(row_dict, capital_cols) if capital_cols else False
            try:
                lon = float(lon_val)
                lat = float(lat_val)
            except (TypeError, ValueError):
                continue
            records.append(
                CityRecord(
                    name=name,
                    iso3=iso3,
                    lon=lon,
                    lat=lat,
                    scalerank=scalerank,
                    pop_max=pop_max,
                    is_capital=is_capital,
                )
            )
        return records

    @staticmethod
    def _to_int_or_none(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _infer_capital(row: dict[str, Any], capital_cols: Sequence[str]) -> bool:
        for col in capital_cols:
            value = row.get(col)
            if value is None:
                continue
            raw = str(value).strip().casefold()
            if raw in {"1", "true", "yes", "y", "admin-0 capital", "admin0cap", "primary"}:
                return True
            if raw in {"0", "false", "no", "n"}:
                continue
            if "capital" in raw and "historic" not in raw:
                return True
        return False

    @staticmethod
    def _require_geopandas() -> Any:
        try:
            import geopandas as gpd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("geopandas is required for Natural Earth data loading") from exc
        return gpd


def _select_best_iso_column(
    dataframe: Any,
    preferred_columns: Sequence[str],
    *,
    iso_allowlist: set[str] | None,
) -> str | None:
    """Pick the best ISO3-like column using schema hints and data-based scoring."""
    existing = [str(col) for col in dataframe.columns]
    by_lower = {col.lower(): col for col in existing}

    candidates: list[str] = []
    for candidate in preferred_columns:
        match = by_lower.get(candidate.lower())
        if match and match not in candidates:
            candidates.append(match)

    for candidate in _heuristic_iso_candidates(existing):
        if candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        return None

    best_col: str | None = None
    best_score: tuple[int, int, int] | None = None
    for candidate in candidates:
        score = _score_iso_values(dataframe[candidate].tolist(), iso_allowlist=iso_allowlist)
        if best_score is None or score > best_score:
            best_col = candidate
            best_score = score

    if best_col is None or best_score is None or best_score[1] == 0:
        return None
    return best_col


def _score_iso_values(
    values: list[Any],
    *,
    iso_allowlist: set[str] | None,
) -> tuple[int, int, int]:
    valid: list[str] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip().upper()
        if len(normalized) == 3 and normalized.isalpha():
            valid.append(normalized)

    valid_set = set(valid)
    overlap_count = len(valid_set & iso_allowlist) if iso_allowlist else 0
    valid_count = len(valid)
    unique_count = len(valid_set)
    return (overlap_count, valid_count, unique_count)


def _heuristic_iso_candidates(columns: Iterable[str]) -> list[str]:
    original = [str(c) for c in columns]
    normalized = [(col, _normalize_column_name(col)) for col in original]
    candidates: list[str] = []
    for original_name, norm in normalized:
        if "A3" not in norm:
            continue
        if any(token in norm for token in ("ISO", "ADM0", "SOV", "WB", "BRK", "GU", "SU")):
            candidates.append(original_name)
    return candidates


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())
