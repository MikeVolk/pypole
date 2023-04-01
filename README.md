# pypole

The "pypole" package is a Python library that provides tools for simulating and fitting magnetic field maps from magnetic dipoles. It includes functions for generating random dipole parameters, calculating magnetic fields, and fitting dipoles to magnetic field maps using least squares regression. The package uses numpy and numba for fast numerical computations, and includes several helper functions for visualizing magnetic field maps and comparing fits.
## Installation

`pypole` uses [Poetry](https://python-poetry.org/) for package management. To install `pypole`, follow these steps:

1. Install Poetry if you haven't already:

```bash
pip install --user poetry
```

Clone the pypole repository from GitHub:

``` bash
git clone https://github.com/MikeVolk/pypole.git
``` 

Navigate to the cloned pypole directory and install the package with Poetry:

```bash
cd pypole
poetry install
```

## Usage

To use the "pypole" package, you can import its modules and call its functions as needed. Here is an example of how to generate a magnetic field map and fit a dipole to it:

```python
import pypole.maps as maps
import pypole.fit as fit

# Generate a magnetic field map
x_grid, y_grid = maps.get_grid(pixels=(100, 100), pixel_size=5e-6)
locations, source_vectors = maps.get_random_sources(n_sources=1)
field_map = maps.calculate_map(x_grid, y_grid, locations, source_vectors)

# Fit a dipole to the magnetic field map
initial_guess = (0, 0, 1e-6, 1e-14, 1e-14, 1e-14)
fit_params = fit.fit_dipole(field_map, initial_guess, pixel_size=5e-6)

# Print the fit parameters
print(fit_params)
```
Documentation
---------------

The "pypole" package includes docstrings and comments for each function, as well as examples of usage. To access the documentation for a specific function, you can use Python's built-in help() function:

## Contributing

If you're interested in contributing to the development of pypole, feel free to create a fork of the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

If you have any questions or suggestions, feel free to contact me:

Email: michaelvolk1979@gmail.com
GitHub: @MikeVolk
