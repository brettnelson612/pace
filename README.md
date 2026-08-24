# PACE
Monorepo to hold PACE (Physics-Aware Coupled Emulator) platform and digital twin implementations.

## Developer workflow

### What Pixi is

Pixi manages PACE's dependencies and environments — one manifest (`pixi.toml`) resolving both conda-forge and PyPI packages together into fully reproducible, per-platform lockfiles. We're on Pixi rather than plain pip/venv because OpenMC is distributed primarily through conda-forge (it ships compiled C++ code and native libraries that pip wheels handle poorly), and Pixi is what lets us manage that alongside our pure-Python dependencies in one place.

Key terms used below:

- **Workspace** — the project-level config at the top of `pixi.toml`: name, authors, channels, and the full set of platforms the repo could ever target.
- **Feature** — a named, reusable bundle of dependencies (e.g. `pace-py`, `openmc`, `cpp-build`, `rust-build`). Inert on its own — only takes effect once included in an environment.
- **Environment** — a named combination of one or more features; this is the actual thing you activate or run commands in (`dev`, `physics`, `default`). An environment's supported platforms are the *intersection* of every feature it includes, not the union — this is how `openmc`'s narrower platform list excludes Apple Silicon from any environment that includes it.
- **Lockfile** (`pixi.lock`) — the fully resolved, exact-version record of every dependency, per environment and per platform. `pixi.toml` expresses intent ("some version of fastapi"); `pixi.lock` records the actual resolved outcome. Commit both together, always.

### Environment setup

1. Install Pixi: `curl -fsSL https://pixi.sh/install.sh | sh`, then restart your shell (or `source ~/.zshrc`).
2. Clone the repo and `cd` into it.
3. Run `pixi install -e dev` to build the local dev environment.
4. Install the [Pixi VS Code extension](https://marketplace.visualstudio.com/items?itemName=renan-r-santos.pixi-code) — or rely on VS Code's built-in Python extension, which auto-detects Pixi environments natively as of mid-2024 and doesn't require a separate install.
5. Open the repo root in VS Code (not a subfolder) — it should auto-detect the `dev` environment. If prompted, select it via `Python: Select Interpreter`.

No `conda activate` needed. Use `pixi shell -e dev` for a subshell with the environment active, or `pixi run -e dev <command>` for one-off commands without a persistent shell.

### Environments

| Environment | Features                                       | Platforms                         | Use for                                                                                     |
| ----------- | ---------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------- |
| `dev`       | `pace-py`                                      | `osx-arm64`, `osx-64`, `linux-64` | Day-to-day service/API work. Native on Apple Silicon, no OpenMC.                            |
| `physics`   | `pace-py`, `openmc`                            | `osx-64`, `linux-64` only         | Anything importing OpenMC or touching the GT pipeline. Runs under Rosetta on Apple Silicon. |
| `default`   | `pace-py`, `openmc`, `cpp-build`, `rust-build` | `osx-64`, `linux-64` only         | Full contributor / CI setup — everything, all languages.                                    |

> **Working on physics/GT code?** `.vscode/settings.json` defaults to the `dev` environment. If your work touches OpenMC or the GT pipeline, run `pixi install -e physics` first, then switch VS Code's interpreter to `.pixi/envs/physics/bin/python` via `Python: Select Interpreter` — `dev` won't have OpenMC and imports will fail. On Apple Silicon, `physics` runs under Rosetta (OpenMC has no native `osx-arm64` build), so expect a one-time Rosetta prompt and somewhat slower execution than `dev`.

### Adding a new dependency

Everything — Python packages included — is declared in `pixi.toml`, never in `src-py/pyproject.toml`. `pyproject.toml` holds only `pace`'s own package identity (name, version, build system); it is not a place to list dependencies.

| Adding...                                                         | Command                                                |
| ----------------------------------------------------------------- | ------------------------------------------------------ |
| Anything available on conda-forge (preferred default)             | `pixi add --feature <feature-name> <package>`          |
| Something PyPI-only, or needing PyPI extras syntax (`pkg[extra]`) | `pixi add --feature <feature-name> --pypi "<package>"` |

Prefer conda-forge unless there's a specific reason not to (a PyPI-only package, or an extras syntax like `[standard]` that only PyPI supports) — that's the whole reason we're on Pixi over plain pip.

After any dependency change, run `pixi install` to confirm the lockfile resolves cleanly, then commit `pixi.toml` and `pixi.lock` together. Lockfile diffs can be large even for a single added package — a new dependency ripples across every environment/platform combination that includes it, so a big diff for a small change is expected, not a red flag.

### Common commands

```bash
pixi install                 # resolve + build environment(s), regenerate lockfile if stale
pixi shell -e dev              # activated subshell for a given environment
pixi run -e dev <command>       # run one command in an environment, no persistent shell
pixi run <task>                  # run a defined task (e.g. `pixi run serve`), from inside an active shell
exit                              # or Ctrl+D — leave a pixi shell
```

`.pixi/` (the local environment cache) is gitignored — never commit it. If your environment ever looks stale or broken, `pixi update` forces a fresh re-solve.
