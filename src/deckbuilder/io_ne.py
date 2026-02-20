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

    COUNTRY_ISO_COLUMNS = ("ADM0_A3", "ISO_A3", "ISO_A3_EH", "ADM0_A3_US", "SOV_A3")
    PLACE_ISO_COLUMNS = ("ADM0_A3", "ISO_A3", "SOV0_A3", "SOV_A3", "GU_A3")
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

    def extract_country_geometry(self, admin0_df: Any, iso3: str) -> Any:
        """Return geometry rows for one ISO3.

        Full polygon normalization (multipolygons, antimeridian handling)
        is deferred to rendering milestones.
        """
        iso_col = _first_existing_column(admin0_df.columns, self.COUNTRY_ISO_COLUMNS)
        if iso_col is None:
            raise ValueError("Could not detect ISO3 column in Natural Earth admin0 data")
        return admin0_df[admin0_df[iso_col] == iso3]

    def extract_city_candidates(
        self,
        places_df: Any,
        *,
        iso3: str,
        capital_fields: Sequence[str],
    ) -> list[CityRecord]:
        """Extract city records for one country from populated places.

        This function already provides deterministic conversion into typed records.
        Additional heuristics can be layered later without changing callers.
        """
        iso_col = _first_existing_column(places_df.columns, self.PLACE_ISO_COLUMNS)
        if iso_col is None:
            raise ValueError("Could not detect country-ISO join column in populated places data")

        name_col = _first_existing_column(places_df.columns, self.PLACE_NAME_COLUMNS)
        lon_col = _first_existing_column(places_df.columns, self.PLACE_LON_COLUMNS)
        lat_col = _first_existing_column(places_df.columns, self.PLACE_LAT_COLUMNS)
        scalerank_col = _first_existing_column(places_df.columns, self.PLACE_SCALERANK_COLUMNS)
        pop_col = _first_existing_column(places_df.columns, self.PLACE_POPMAX_COLUMNS)

        if name_col is None or lon_col is None or lat_col is None:
            raise ValueError("Populated places dataset missing required name/lon/lat columns")

        capital_cols: list[str] = []
        for field in capital_fields:
            col = _first_existing_column(places_df.columns, [field])
            if col:
                capital_cols.append(col)

        subset = places_df[places_df[iso_col] == iso3]
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
            is_capital = self._infer_capital(row_dict, capital_cols)
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
