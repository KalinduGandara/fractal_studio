"""Escape-time kernels (numba, parallel over rows).

Every kernel returns a float64 image of *smooth iteration counts* ("nu"):
  nu >= 0  -> escaped; continuous (band-free) iteration count
  nu <  0  -> did not escape within ``max_iter`` (treated as interior)

The view is described by a center, the complex span of the *shorter* image
side, and a rotation angle. Pixel (row, col) maps to

    c = center + rot * ((col - w/2 + 0.5) * px  -  i * (row - h/2 + 0.5) * px)

so the image's "up" is +imag (after rotation). Kernels that want an upside-
down convention (Burning Ship) flip the sign of the imaginary offset.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

BAILOUT = 256.0  # escape radius; large radius => smoother continuous colouring
_B2 = BAILOUT * BAILOUT
_LOG2 = math.log(2.0)
_LOG_B = math.log(BAILOUT)
INTERIOR = -1.0


@njit(cache=True, inline="always")
def _smooth(n: int, zr2_zi2: float) -> float:
    # nu = n + 1 - log2(log|z| / log B)  (normalised so nu is continuous in n)
    log_zn = 0.5 * math.log(zr2_zi2)
    return n + 1.0 - math.log(log_zn / _LOG_B) / _LOG2


@njit(cache=True, fastmath=False)
def mandelbrot_point(cr: float, ci: float, max_iter: int) -> float:
    """Smooth iteration count for one point of the Mandelbrot set."""
    # Quick interior tests: main cardioid and period-2 bulb.
    q = (cr - 0.25) ** 2 + ci * ci
    if q * (q + (cr - 0.25)) <= 0.25 * ci * ci:
        return INTERIOR
    if (cr + 1.0) ** 2 + ci * ci <= 0.0625:
        return INTERIOR
    zr = 0.0
    zi = 0.0
    # Brent-style periodicity check: interior points settle onto an exact
    # floating point cycle; detecting it saves running to max_iter.
    pr = 0.0
    pi = 0.0
    check = 8
    for n in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > _B2:
            return _smooth(n, zr2 + zi2)
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
        if zr == pr and zi == pi:
            return INTERIOR
        if n == check:
            pr = zr
            pi = zi
            check *= 2
    return INTERIOR


@njit(cache=True, fastmath=False)
def burning_ship_point(cr: float, ci: float, max_iter: int) -> float:
    """Smooth iteration count for the Burning Ship: z -> (|Re z| + i|Im z|)^2 + c."""
    zr = 0.0
    zi = 0.0
    pr = 0.0
    pi = 0.0
    check = 8
    for n in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > _B2:
            return _smooth(n, zr2 + zi2)
        zi = 2.0 * abs(zr * zi) + ci
        zr = zr2 - zi2 + cr
        if zr == pr and zi == pi:
            return INTERIOR
        if n == check:
            pr = zr
            pi = zi
            check *= 2
    return INTERIOR


@njit(cache=True, fastmath=False)
def julia_point(zr: float, zi: float, cr: float, ci: float, max_iter: int) -> float:
    """Smooth iteration count for the filled Julia set of z^2 + c."""
    pr = 0.0
    pi = 0.0
    check = 8
    for n in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > _B2:
            return _smooth(n, zr2 + zi2)
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
        if zr == pr and zi == pi:
            return INTERIOR
        if n == check:
            pr = zr
            pi = zi
            check *= 2
    return INTERIOR


KIND_MANDELBROT = 0
KIND_BURNING_SHIP = 1
KIND_JULIA = 2


@njit(cache=True, parallel=True)
def render_nu(
    kind: int,
    cx: float,
    cy: float,
    span: float,
    angle: float,
    width: int,
    height: int,
    max_iter: int,
    jr: float,
    ji: float,
) -> np.ndarray:
    """Render a smooth-iteration image of shape (height, width).

    ``span`` is the complex extent of the shorter image side. ``jr, ji`` is
    the Julia parameter (ignored by the other kinds).
    """
    out = np.empty((height, width), dtype=np.float64)
    px = span / min(width, height)
    ca = math.cos(angle)
    sa = math.sin(angle)
    flip = -1.0 if kind == KIND_BURNING_SHIP else 1.0
    for row in prange(height):
        dy = -(row - height * 0.5 + 0.5) * px * flip
        for col in range(width):
            dx = (col - width * 0.5 + 0.5) * px
            # rotate the offset, then add to the (high-magnitude) center
            ox = dx * ca - dy * sa
            oy = dx * sa + dy * ca
            x = cx + ox
            y = cy + oy
            if kind == KIND_MANDELBROT:
                out[row, col] = mandelbrot_point(x, y, max_iter)
            elif kind == KIND_BURNING_SHIP:
                out[row, col] = burning_ship_point(x, y, max_iter)
            else:
                out[row, col] = julia_point(x, y, jr, ji, max_iter)
    return out
