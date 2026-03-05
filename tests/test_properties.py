"""Property-based tests using hypothesis."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pypole import compute, maps
from pypole.convert import dim2xyz, xyz2dim


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def valid_dim_arrays(draw):
    """DIM arrays: (declination [0,360), inclination (-85,85), magnitude > 0).

    Poles (inc = +-90) are excluded because declination is undefined there,
    making the roundtrip ill-defined.  Zero magnitude is excluded to avoid
    division by zero in xyz2dim.
    """
    n = draw(st.integers(1, 20))
    dec = draw(
        arrays(
            np.float64,
            shape=(n,),
            # Avoid near-zero angles: sin(dec) ~ 1e-16 at dec=1e-14 deg,
            # which exhausts float64 precision in the roundtrip.
            elements=st.floats(1.0, 359.0, allow_nan=False, allow_infinity=False),
        )
    )
    inc = draw(
        arrays(
            np.float64,
            shape=(n,),
            elements=st.floats(-85.0, 85.0, allow_nan=False, allow_infinity=False),
        )
    )
    mag = draw(
        arrays(
            np.float64,
            shape=(n,),
            elements=st.floats(1e-20, 1e-10, allow_nan=False, allow_infinity=False),
        )
    )
    return np.column_stack([dec, inc, mag])


def field_map_arrays(min_val=-1e-6, max_val=1e-6, min_side=2, max_side=30):
    """Strategy for 2D float64 field maps."""
    return arrays(
        np.float64,
        shape=st.tuples(
            st.integers(min_side, max_side), st.integers(min_side, max_side)
        ),
        elements=st.floats(min_val, max_val, allow_nan=False, allow_infinity=False),
    )


# ---------------------------------------------------------------------------
# convert: dim2xyz / xyz2dim roundtrip
# ---------------------------------------------------------------------------


@given(dim=valid_dim_arrays())
def test_dim_xyz_dim_roundtrip(dim):
    """dim -> xyz -> dim recovers inclination and magnitude exactly,
    and declination modulo 360."""
    xyz = dim2xyz(dim)
    recovered = xyz2dim(xyz)

    np.testing.assert_allclose(recovered[:, 2], dim[:, 2], rtol=1e-10)  # magnitude
    np.testing.assert_allclose(recovered[:, 1], dim[:, 1], atol=1e-8)  # inclination

    dec_diff = np.abs(recovered[:, 0] - dim[:, 0]) % 360
    # wrap differences > 180 back (e.g. 359.9 vs 0.1)
    dec_diff = np.minimum(dec_diff, 360 - dec_diff)
    assert np.all(dec_diff < 1e-6), f"declination roundtrip failed: {dec_diff}"


@given(dim=valid_dim_arrays())
def test_xyz_dim_xyz_roundtrip(dim):
    """dim -> xyz -> dim -> xyz recovers the xyz vector."""
    xyz = dim2xyz(dim)
    recovered_dim = xyz2dim(xyz)
    recovered_xyz = dim2xyz(recovered_dim)

    np.testing.assert_allclose(recovered_xyz, xyz, rtol=1e-9, atol=1e-30)


# ---------------------------------------------------------------------------
# compute: rms
# ---------------------------------------------------------------------------


@given(field_map=field_map_arrays())
def test_rms_non_negative(field_map):
    """RMS of any map is >= 0."""
    assert compute.rms(field_map) >= 0


@given(
    field_map=arrays(
        np.float64,
        shape=st.tuples(st.integers(2, 20), st.integers(2, 20)),
        elements=st.floats(-1e-6, 1e-6, allow_nan=False, allow_infinity=False),
    ),
    scale=st.floats(0.5, 2.0, allow_nan=False, allow_infinity=False),
)
def test_rms_scales_linearly(field_map, scale):
    """rms(scale * x) == abs(scale) * rms(x)."""
    np.testing.assert_allclose(
        compute.rms(scale * field_map), abs(scale) * compute.rms(field_map), rtol=1e-10
    )


# ---------------------------------------------------------------------------
# compute: dipolarity_param
# ---------------------------------------------------------------------------


@given(field_map=field_map_arrays())
def test_dipolarity_perfect_fit_is_one(field_map):
    """dipolarity_param(x, x) == 1 for any map whose rms doesn't underflow."""
    # Values near subnormal (~1e-273) square to 0 in float64, so rms=0 -> nan.
    assume(compute.rms(field_map) > 0)
    dp = compute.dipolarity_param(field_map, field_map)
    np.testing.assert_allclose(dp, 1.0, atol=1e-10)


@given(field_map=field_map_arrays(), fitted_map=field_map_arrays())
def test_dipolarity_shape_invariant(field_map, fitted_map):
    """dipolarity_param raises ValueError on shape mismatch, returns scalar otherwise."""
    if field_map.shape != fitted_map.shape:
        with pytest.raises(ValueError):
            compute.dipolarity_param(field_map, fitted_map)
    else:
        result = compute.dipolarity_param(field_map, fitted_map)
        assert np.isscalar(result) or result.ndim == 0


# ---------------------------------------------------------------------------
# compute: pad_map
# ---------------------------------------------------------------------------


@given(
    shape=st.tuples(st.integers(1, 20), st.integers(1, 20)),
    oversample=st.integers(2, 5),
)
def test_pad_map_output_shape(shape, oversample):
    """Padded map has shape original * (1 + 2*(oversample-1)) on last two axes."""
    field_map = np.zeros(shape, dtype=np.float64)
    padded = compute.pad_map(field_map, oversample)
    factor = 1 + 2 * (oversample - 1)
    assert padded.shape == (shape[0] * factor, shape[1] * factor)


@given(
    arr=arrays(
        np.float64,
        shape=st.tuples(st.integers(1, 10), st.integers(1, 10)),
        elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    ),
    oversample=st.integers(2, 4),
)
def test_pad_map_preserves_data(arr, oversample):
    """Original data is preserved in the center of the padded map."""
    padded = compute.pad_map(arr, oversample)
    ypix, xpix = arr.shape
    row0 = (padded.shape[0] - ypix) // 2
    col0 = (padded.shape[1] - xpix) // 2
    np.testing.assert_array_equal(padded[row0 : row0 + ypix, col0 : col0 + xpix], arr)


# ---------------------------------------------------------------------------
# maps: get_grid
# ---------------------------------------------------------------------------


@given(
    pixels=st.tuples(st.integers(2, 100), st.integers(2, 100)),
    pixel_size=st.floats(1e-9, 1e-3, allow_nan=False, allow_infinity=False),
)
def test_get_grid_shape(pixels, pixel_size):
    """get_grid returns arrays of shape (pixels[1], pixels[0])."""
    x, y = maps.get_grid(pixels, pixel_size)
    assert x.shape == (pixels[1], pixels[0])
    assert y.shape == (pixels[1], pixels[0])
    assert x.dtype == np.float64
    assert y.dtype == np.float64


@given(
    pixels=st.tuples(st.integers(2, 50), st.integers(2, 50)),
    pixel_size=st.floats(1e-9, 1e-3, allow_nan=False, allow_infinity=False),
)
def test_get_grid_symmetric(pixels, pixel_size):
    """Grid is centred: max == -min along each axis."""
    x, y = maps.get_grid(pixels, pixel_size)
    np.testing.assert_allclose(x.max(), -x.min(), rtol=1e-10)
    np.testing.assert_allclose(y.max(), -y.min(), rtol=1e-10)
