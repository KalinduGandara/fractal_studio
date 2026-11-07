import math

import numpy as np
import pytest

from fractal_studio import kernels
from fractal_studio.refine import refine_misiurewicz, refine_nucleus

INTERIOR = kernels.INTERIOR


@pytest.mark.parametrize("c", [0j, -1 + 0j, -0.1 + 0.1j, -1.75487766624669276 + 0j, 0.25 + 0j])
def test_mandelbrot_interior_points(c):
    assert kernels.mandelbrot_point(c.real, c.imag, 2000) == INTERIOR


@pytest.mark.parametrize("c", [1 + 0j, 0.5 + 0.5j, -2.1 + 0j, 0.3 + 0j])
def test_mandelbrot_exterior_points(c):
    assert kernels.mandelbrot_point(c.real, c.imag, 2000) > 0


def test_smooth_count_matches_reference():
    # independent pure-python smooth iteration count
    c = complex(0.37, 0.42)
    z, n = 0j, 0
    while abs(z) <= kernels.BAILOUT and n < 1000:
        z = z * z + c
        n += 1
    assert n < 1000
    ref = n + 1 - math.log(math.log(abs(z)) / math.log(kernels.BAILOUT)) / math.log(2)
    assert kernels.mandelbrot_point(c.real, c.imag, 1000) == pytest.approx(ref, abs=1e-9)


def test_smooth_count_is_continuous():
    xs = np.linspace(0.60, 0.70, 200)  # smooth exterior region, crosses several bands
    vals = np.array([kernels.mandelbrot_point(x, 0.3, 1000) for x in xs])
    assert np.all(vals > 0)
    assert np.max(np.abs(np.diff(vals))) < 0.5  # no integer jumps (banding)


def test_periodicity_check_does_not_change_exterior():
    # a frame near the boundary: every escaped value must equal a brute-force count
    nu = kernels.render_nu(kernels.KIND_MANDELBROT, -0.745, 0.11, 0.02, 0.0, 24, 24, 3000, 0.0, 0.0)
    rng = np.random.default_rng(0)
    for r, c in rng.integers(0, 24, size=(40, 2)):
        px = 0.02 / 24
        x = -0.745 + (c - 12 + 0.5) * px
        y = 0.11 - (r - 12 + 0.5) * px
        z, n = 0j, 0
        cc = complex(x, y)
        while abs(z) <= kernels.BAILOUT and n < 3000:
            z = z * z + cc
            n += 1
        if n < 3000:
            assert nu[r, c] > 0
            assert nu[r, c] == pytest.approx(
                n + 1 - math.log(math.log(abs(z)) / math.log(kernels.BAILOUT)) / math.log(2),
                abs=1e-6,
            )


def test_burning_ship_known_points():
    assert kernels.burning_ship_point(0.0, 0.0, 500) == INTERIOR
    assert kernels.burning_ship_point(-1.75, 0.0, 500) == INTERIOR  # real axis = Mandelbrot
    assert kernels.burning_ship_point(1.0, 1.0, 500) > 0
    # not symmetric in Im(c), unlike the Mandelbrot set
    a = kernels.burning_ship_point(-0.5, 0.6, 500)
    b = kernels.burning_ship_point(-0.5, -0.6, 500)
    assert a != b


def test_julia_known_points():
    # c = 0: the Julia set is the unit circle
    assert kernels.julia_point(0.5, 0.0, 0.0, 0.0, 500) == INTERIOR
    assert kernels.julia_point(1.5, 0.0, 0.0, 0.0, 500) > 0


def test_render_is_deterministic_and_shaped():
    args = (kernels.KIND_MANDELBROT, -0.5, 0.0, 3.0, 0.3, 64, 48, 300, 0.0, 0.0)
    a = kernels.render_nu(*args)
    b = kernels.render_nu(*args)
    assert a.shape == (48, 64)
    assert np.array_equal(a, b)


def test_refined_landmarks_are_exact():
    m = refine_misiurewicz(-0.10109636 + 0.95628651j, 4, 1)
    z = 0j
    orbit = []
    for _ in range(6):
        z = z * z + m
        orbit.append(z)
    assert abs(orbit[4] - orbit[3]) < 1e-12  # pre-periodic: f^5(0) == f^4(0)
    n = refine_nucleus(-1.75 + 0j, 3)
    z = 0j
    for _ in range(3):
        z = z * z + n
    assert abs(z) < 1e-14
    assert n.real == pytest.approx(-1.7548776662466927, abs=1e-15)
