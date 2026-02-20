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
python -m deckbuilder build --config config.yaml
```

Or using console script:

```bash
deckbuilder validate --config config.yaml
deckbuilder build --config config.yaml
```

## Current Status

- Implemented:
  - Typed config loading and validation
  - Domain models and overrides parsing
  - Manual capital fallback override support for countries missing populated-places capital entries
  - CLI command structure (`build`, `validate`, `render-maps`, `fetch-flags`, `build-deck`)
  - Milestone 1 validation: Natural Earth loading, UN ISO3 admin0 coverage checks, geometry cardinality checks, and capital coverage checks (with overrides)
  - Build directory/bootstrap, logging, manifest writing
  - QA index skeleton generation
- Stubbed (planned milestones):
  - Deterministic map rendering
  - Wikimedia flag retrieval and attribution harvesting
  - `genanki` deck packaging

## Design Principles

- Deterministic outputs from pinned inputs
- Strict separation of concerns by module
- Fail-loud validation in strict mode
- Config-driven behavior (no hidden constants)
