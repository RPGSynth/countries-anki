# countries-anki

Automated, reproducible pipeline skeleton for generating an Anki deck with one note per UN member state:
- Front: country flag only
- Back: country name + rendered map with main cities

This repository provides a production-grade architecture and working Milestones 3, 4, and 5 (map rendering, flag fetching, and deck packaging).

## Layout

```text
countries-anki/
  pyproject.toml
  README.md
  config.yaml
  data/
    un_members.yaml
    cities_overrides.yaml
    flags_overrides.yaml
    render_overrides.yaml
    natural_earth/
  src/
    deckbuilder/
      __init__.py
      cli.py
      config.py
      models.py
      io_ne.py
      countries.py
      cities.py
      inspect_report.py
      render.py
      flags.py
      anki.py
      validate.py
      qa.py
      util.py
  build/
    media/maps/
    media/flags/
    attribution/
    qa/thumbs/
    deck/
    manifests/
    logs/
```

## Quick Start

1. Create a virtual environment and install dependencies.
2. Put Natural Earth datasets in `data/natural_earth/`.
3. Run:

```bash
python -m pip install -e .
python -m deckbuilder validate --config config.yaml
python -m deckbuilder select-cities --config config.yaml
python -m deckbuilder render-maps --config config.yaml
python -m deckbuilder fetch-flags --config config.yaml
python -m deckbuilder build-deck --config config.yaml
python -m deckbuilder build --config config.yaml
python -m deckbuilder inspect --config config.yaml --limit 40
```

Or using console script:

```bash
deckbuilder validate --config config.yaml
deckbuilder select-cities --config config.yaml
deckbuilder render-maps --config config.yaml
deckbuilder fetch-flags --config config.yaml
deckbuilder build-deck --config config.yaml
deckbuilder render-maps --config config.yaml --debug-render --limit-countries 3
deckbuilder render-maps --config config.yaml --debug-render --country BEL --country CAN --country MCO --clean-maps
deckbuilder build --config config.yaml
deckbuilder inspect --config config.yaml --country FRA --country CIV
```

Open `build/qa/index.html` to visually review generated maps.

## Current Status

  - Implemented:
  - Typed config loading and validation
  - Canonical capital source from `data/un_members.yaml` (`capital` field per country)
  - Automatic capital relabel fallback: if canonical capital name is missing in populated places, select a suspected capital city in-country and relabel it to the canonical name
  - Domain models and overrides parsing
  - Manual capital fallback override support for countries missing populated-places capital entries
  - Milestone 2 deterministic city selection pipeline (`select-cities`) with:
    - tiny-country automatic N reduction (`min_n_for_tiny`)
    - override-aware capital/city resolution
    - per-country JSON audit output at `build/manifests/city_selection.json`
  - Milestone 3 deterministic map rendering (`render-maps`) with:
    - fixed-size PNG output to `build/media/maps/map_{ISO3}.png`
    - admin0 geometry merge per country (multi-row dissolve/union for rendering)
    - country-adaptive projection (reduces high-latitude horizontal stretching)
    - adaptive tiny-country/inset extents so microstates render larger in frame
    - surrounding-country boundary context as dotted outlines (no extra city labels)
    - configurable background mode: `white`, `google_flat` (default), `satellite`
    - real tile basemaps (no-label flat provider for city-overlay clarity)
    - city markers + capital marker styling from `config.yaml`
    - deterministic label placement with collision dropping (capital never dropped)
    - per-country render overrides via `data/render_overrides.yaml`:
      - `outline` (force country outline on/off)
      - `center_on_capital` (force capital-centered viewport on/off)
      - `inset` (force insets on/off)
  - Milestone 4 Wikimedia flag pipeline (`fetch-flags`) with:
    - override support via `data/flags_overrides.yaml`
    - deterministic ISO3-based title resolution from Wikidata with fallback probes
    - SVG preference with fallback to Wikimedia-rendered PNG thumbnails
    - normalized output to `build/media/flags/flag_{ISO3}.png`
    - attribution export to `build/attribution/flags.json`
  - Milestone 5 Anki deck packaging (`build-deck`) with:
    - one deterministic note per UN member
    - stable `deck_id` / `model_id` from config
    - media embedding for all `map_{ISO3}.png` and `flag_{ISO3}.png`
    - strict missing-media failure before packaging
    - output `.apkg` at `project.output_apkg`
  - CLI command structure (`build`, `validate`, `select-cities`, `inspect`, `render-maps`, `fetch-flags`, `build-deck`)
  - Milestone 1 validation: Natural Earth loading, UN ISO3 admin0 coverage checks, geometry cardinality checks, and capital coverage checks (with overrides)
  - Inspection report command: visual HTML + JSON diagnostics for joins, city selection outputs, and map availability
  - Build directory/bootstrap, logging, manifest writing
  - QA index generation with per-country map/flag status and previews
  - `build` now runs validation + city-selection + map rendering + flag fetching + deck packaging

## Design Principles

- Deterministic outputs from pinned inputs
- Strict separation of concerns by module
- Fail-loud validation in strict mode
- Config-driven behavior (no hidden constants)
