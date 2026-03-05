# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                         # install all dependencies
uv run pytest                                   # run all tests (includes doctests)
uv run pytest tests/test_compute.py::test_rms  # run a single test
make codestyle                                  # format with ruff (format + fix)
make check-codestyle                            # lint check without modifying
make ty                                         # type-check with ty
make lint                                       # test + check-codestyle + ty
make pre-commit                                 # run all pre-commit hooks
```

### Tooling
- **uv** — package manager; `uv sync` installs all deps including dev group
- **ruff** — linter + formatter (replaces black, isort, pyupgrade, pylint); config in `[tool.ruff]`
- **ty** — type checker (replaces mypy); run via `make ty` or pre-commit local hook
- **loguru** — logging (`from loguru import logger`); use `{}` placeholders, never f-strings in log calls
- **pre-commit** hooks: `pre-commit-hooks` v6.0.0, `ruff-pre-commit` v0.15.4, `ty` (local)

Pytest runs with `--doctest-modules`, so doctests in source files are part of the test suite. Run from the repo root — `test_compute.py` loads `tests/QDMlab_comparison_uc.mat` via a relative path.

## Architecture

`pypole` simulates and fits magnetic field maps from magnetic dipoles. All arrays use `NDArray64 = NDArray[np.float64]` defined in `__init__.py`.

### Module roles

| Module | Role |
|--------|------|
| `dipole.py` | Core physics: `dipole_field()` computes Bz from a single dipole (units: T). Also hosts `calculate_map()` and `synthetic_map()`. |
| `maps.py` | Grid and random source generation: `get_grid()`, `get_random_sources()`, `get_random_dim()`, `get_random_locations()`. |
| `compute.py` | Signal processing: FFT-based `upward_continue()`, vectorized `rms()`, `dipolarity_param()`, `pad_map()`. |
| `fit.py` | Dipole fitting: `fit_dipole()` wraps `_fit_dipole()` which uses `scipy.optimize.least_squares` with Huber loss + TRF method. |
| `convert.py` | Coordinate conversion between Cartesian (x,y,z) and geomagnetic polar (declination, inclination, magnitude). |
| `plotting.py` | Matplotlib helpers for field maps and fit results. |

### Numba usage
- `dipole_field()`: `@njit(fastmath=True, parallel=True)` — pixel-level parallelism
- `calculate_map()`: `@jit(parallel=True, fastmath=True)` — parallelises over sources with `prange`
- `rms()`: `@guvectorize` — operates on arbitrary leading batch dims `(..., m, n) -> (...,)`
- `dim2xyz()` / `xyz2dim()`: `@guvectorize` — element-wise on `(n,) -> (n,)`

Expect a JIT warm-up delay on first call. Do not add Python-level branching inside `@njit` functions.

ty does not understand numba internals — guvectorize output array assignments and `prange` are suppressed with `# ty: ignore[...]` comments. Do not remove these.

### Key conventions
- **Dipole parameters** are always 6-tuples ordered `(x_source, y_source, z_source, mx, my, mz)` — positions in metres, moments in Am².
- **Coordinate system** (`convert.py`): North = -Y axis, +Z points down (NED). Declination measured CCW from North in X-Y plane; 0° = -Y, 90° = +X. Inclination > 0 means pointing downward (+Z direction), matching pmagpy / standard NED convention. `inc = +90` → `z = +mz` (down); `inc = -90` → `z = -mz` (up).
- **`get_grid`** (`maps.py`): spacing is exactly `pixel_size`; grid is centred at 0. Formula: `linspace(-(N-1)/2, (N-1)/2, N) * pixel_size`.
- **`calculate_map`** lives in `dipole.py`, not `maps.py`. It sums contributions from all sources; `sensor_distance` is added to the source z-coordinate before field evaluation.
- **`fit_dipole`** returns `scipy.optimize.OptimizeResult` — access optimised parameters via `.x`, not as a tuple.
- **Git workflow**: gitflow — feature branches off `develop`, merge back to `develop`, release to `master`.
- **Commit attribution**: never add `Co-Authored-By: Claude` or any AI attribution lines to commit messages.
