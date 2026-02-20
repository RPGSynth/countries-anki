# countries-anki

Automated, reproducible pipeline skeleton for generating an Anki deck with one note per UN member state:
- Front: country name + rendered map with main cities
- Back: country flag

This repository currently provides a production-grade architecture and module boundaries. Core business logic (map rendering, Wikimedia fetch, deck packaging) is intentionally stubbed for iterative implementation.

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
python -m deckbuilder build --config config.yaml
python -m deckbuilder inspect --config config.yaml --limit 40
```

Or using console script:

```bash
deckbuilder validate --config config.yaml
deckbuilder select-cities --config config.yaml
deckbuilder build --config config.yaml
deckbuilder inspect --config config.yaml --country FRA --country CIV
```

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
  - CLI command structure (`build`, `validate`, `select-cities`, `inspect`, `render-maps`, `fetch-flags`, `build-deck`)
  - Milestone 1 validation: Natural Earth loading, UN ISO3 admin0 coverage checks, geometry cardinality checks, and capital coverage checks (with overrides)
  - Inspection report command: visual HTML + JSON diagnostics for joins and city selection outputs
  - Build directory/bootstrap, logging, manifest writing
  - QA index skeleton generation
  - `build` now runs validation + city-selection audit before downstream stub steps
- Stubbed (planned milestones):
  - Deterministic map rendering
  - Wikimedia flag retrieval and attribution harvesting
  - `genanki` deck packaging

## Design Principles

- Deterministic outputs from pinned inputs
- Strict separation of concerns by module
- Fail-loud validation in strict mode
- Config-driven behavior (no hidden constants)
