# PACE
[![CI](https://img.shields.io/github/actions/workflow/status/brettnelson612/pace/ci.yml?branch=main&label=CI&logo=github)](https://github.com/brettnelson612/pace/actions/workflows/ci.yml)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/brettnelson612/pace/main/python/pyproject.toml)](python/pyproject.toml)
[![License](https://img.shields.io/github/license/brettnelson612/pace)](LICENSE)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**PACE (Physics-Aware Coupled Emulator)** is a reactor modeling platform that sits above open-source physics solvers. PACE will allow a user to build a versioned, composable model, then turn that model into inputs for OpenMC, MOOSE and Cardinal. Once a model is complete, PACE will allow a user to run physics simulations + V&V and turn the resulting output into datasets and surrogate models.

PACE does not solve physics or couple solvers. Transport, heat conduction, fluid flow, etc belong to the solvers; field transfer and Picard iteration belong to Cardinal. PACE owns everything around a simulation: defining the model, versioning it, generating solver inputs, launching runs, and building datasets, surrogates and uncertainty quantification on the results.

## PACE Scope

- **Reactor models.** Geometry, materials and a composition hierarchy: pellets → rods → pin cells → lattices → assemblies → cores. Each model holds the minimum complete input the solvers need.
- **Versioning and provenance.** Every model object is versioned. A version derived from a simulation result records which ground-truth (GT) run produced it.
- **Solver input generation.**
  - OpenMC models (neutronics).
  - MOOSE inputs, through Reactor module mesh generators: Heat Transfer for fuel and cladding, THM for the coolant.
  - Cardinal inputs for coupled runs.
- **Ground Truth run management.** Launch, track and store runs, each linked to the model version it ran.
- **Datasets, surrogates and UQ.** Build datasets from GT runs, train surrogates on them, and quantify uncertainty across models, runs and surrogates.
- **Workshop UI.** Design, browse and edit reactor models; manage each reactor's runs, datasets and surrogates.

**Validation:** initial POC models will be LWR's to run VERA problems [VERA core physics benchmark progression problems](https://corephysics.com/docs/CASL-U-2012-0131-004.pdf). More may be added from there.

**Non-goals:**
- writing solvers or coupling algorithms;
- targeting proprietary or export-controlled codes (MCNP, BISON, Griffin) as primary solvers, so that results stay reproducible by anyone.

### Roadmap to v1

1. **PACE → OpenMC.** Reproduce VERA problems 1 and 2.
2. **Coupled pin cell.** OpenMC + MOOSE Heat Transfer + THM, coupled through Cardinal.

### Roadmap to v2+
1. **Parametric studies.**
   - persisted GT runs, model variants and batch runs;
   - datasets and surrogates, with hooks for the UQ layer;
   - fast reactor modeling (lots of work needed here, and V&V datasets are limited/restricted).
2. **Service and UI.**
   - Providing a "Reactor Workshop" to make design/visualization easier
   - Providing a Reactors hub to compile everything a user cares about regarding a reactor (GT runs, datasets, surrogates, etc.)

### Knowledge Base / Resources to get us there
[PACE Knowledge Base](https://brettnelson612.github.io/pace/knowledge-base.html) — a map of the different knowledge domains involved here along with helpful resources + anecdotes.

## Repository layout

```
python/
  src/pace/
    core/       domain model: geometry, material, components, lattices, blueprints
    db/         persistence: SQLite registry, DAOs, reference index, object store
    services/   registry rules: registration, validation, hydration
    solvers/    solver translators (OpenMC, MOOSE, Cardinal) — in progress
  tests/        unit and integration tests
cpp/, rust/     reserved for compiled components
docs/           project knowledge base
```

## Development

PACE uses [Pixi](https://pixi.prefix.dev/) for dependencies, environments and tasks. Pixi is used instead of pip/venv because:

- OpenMC and MOOSE are distributed through conda channels;
- PACE's pure-Python dependencies come from PyPI;
- Pixi resolves both together into one lockfile, per platform.

`pixi.toml` declares what we want; `pixi.lock` records exactly what was resolved. Commit the two together.

### Setup

1. Install Pixi: `curl -fsSL https://pixi.sh/install.sh | sh`, then restart your shell.
2. Clone the repo and run `pixi install -e no-phys-dev`.
3. Open the repo root in VS Code. `.vscode/settings.json` points the Python interpreter at the `no-phys-dev` environment.

### Environments

| Environment | Adds | Platforms | Use for |
| --- | --- | --- | --- |
| `no-phys-dev` | dev tools, `pace-py` | `osx-arm64`, `osx-64`, `linux-64` | Everyday modeling, registry and service work. The only environment that runs natively on Apple Silicon. |
| `py-phys-dev` | + OpenMC | `osx-64`, `linux-64` | Anything that imports OpenMC. Runs under Rosetta on Apple Silicon. |
| `full-phys-dev` | + MOOSE, C++/Rust toolchains | `osx-64`, `linux-64` | Cardinal / MOOSE work. The only environment with the `moose-*` tasks. |

### Common commands

```bash
pixi shell -e no-phys-dev                 # shell with the environment active
pixi run -e no-phys-dev check             # everything CI runs: ruff, black, mypy, tests
pixi run -e no-phys-dev format            # ruff --fix + black
pixi add --feature <feature> <package>    # add a conda-forge dependency (preferred)
pixi add --feature <feature> --pypi <pkg> # add a PyPI-only dependency
```

Dependencies go in `pixi.toml`, never in `python/pyproject.toml`; `pyproject.toml` only defines the `pace` package itself. After changing dependencies, run `pixi install` and commit `pixi.toml` and `pixi.lock` together.

## License

See [LICENSE](LICENSE).
