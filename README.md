# pypole

`pypole` is a Python library for simulating and fitting magnetic field maps from magnetic dipole sources. It provides tools for generating synthetic dipole fields, fitting dipoles to measured maps, and processing magnetic field data using FFT-based methods.

Uses NumPy and Numba for fast numerical computation, with JIT-compiled kernels for pixel-level parallelism.

## Installation

`pypole` uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
git clone https://github.com/MikeVolk/pypole.git
cd pypole
uv sync
```

## Quick Start

```python
import numpy as np
from pypole import maps, dipole, fit

# Generate observation grid (100x100 pixels, 1 um pixel size)
x_grid, y_grid = maps.get_grid(pixels=(100, 100), pixel_size=1e-6)

# Generate a random single-dipole source
locations, moments = maps.get_random_sources(n_sources=1)

# Compute the Bz field map (sensor 5 um above sample)
field_map = dipole.calculate_map(x_grid, y_grid, locations, moments, sensor_distance=5e-6)

# Fit a single dipole to the map
p0 = (*locations[0], *moments[0])  # initial guess: (x, y, z, mx, my, mz)
result = fit.fit_dipole(field_map, p0, pixel_size=1e-6)

# Access optimised parameters
print(result.x)   # (x_source, y_source, z_source, mx, my, mz)
print(result.success)
```

## Modules

| Module | Role |
|--------|------|
| `dipole` | Core physics: `dipole_field()` computes Bz from a single dipole (T). `calculate_map()` sums contributions from all sources. `synthetic_map()` generates a complete random map. |
| `maps` | Grid and source generation: `get_grid()`, `get_random_sources()`, `get_random_dim()`, `get_random_locations()`. |
| `compute` | Signal processing: FFT-based `upward_continue()`, vectorized `rms()`, `dipolarity_param()`, `pad_map()`. |
| `fit` | Dipole fitting: `fit_dipole()` uses `scipy.optimize.least_squares` with Huber loss + TRF method. `fit_dipole_n_maps()` fits a batch of maps in parallel via Numba. |
| `convert` | Coordinate conversion between Cartesian (x, y, z) and geomagnetic polar (declination, inclination, magnitude). |
| `plotting` | Matplotlib helpers for field maps and fit results. |

## Conventions

- **Dipole parameters** are always 6-tuples ordered `(x_source, y_source, z_source, mx, my, mz)` — positions in metres, moments in Am².
- **Coordinate system**: NED (North-East-Down). North = -Y axis, +Z points down. Declination is measured CCW from North; 0° = -Y, 90° = +X. Inclination > 0 means pointing downward (+Z).
- **`fit_dipole`** returns a `scipy.optimize.OptimizeResult` — access optimised parameters via `.x`.

## Signal Processing

```python
from pypole import compute

# Upward continue a map by 2 um
continued = compute.upward_continue(field_map, distance=2e-6, pixel_size=1e-6)

# Compute RMS (works on 2D or batched 3D arrays)
rms_value = compute.rms(field_map)

# Compute dipolarity parameter (1 = perfect dipole)
dp = compute.dipolarity_param(field_map, fitted_map)
```

## Coordinate Conversion

```python
from pypole import convert
import numpy as np

# Convert declination/inclination/moment -> x/y/z
dim = np.array([[45.0, 30.0, 1e-14]])   # (dec deg, inc deg, moment Am^2)
xyz = np.zeros((1, 3))
convert.dim2xyz(dim, xyz)

# Convert x/y/z -> declination/inclination/moment
xyz_in = np.array([[1e-14, 0.0, 0.0]])
dim_out = np.zeros((1, 3))
convert.xyz2dim(xyz_in, dim_out)
```

## Development

```bash
uv sync                   # install all dependencies (including dev group)
uv run pytest             # run all tests + doctests
make codestyle            # format with ruff (format + fix)
make check-codestyle      # lint check without modifying
make ty                   # type-check with ty
make lint                 # test + check-codestyle + ty
make pre-commit           # run all pre-commit hooks
```

## License

MIT License. See the LICENSE file for details.

## Contact

- Email: michaelvolk1979@gmail.com
- GitHub: [@MikeVolk](https://github.com/MikeVolk)
