"""Newton refinement of exact Mandelbrot landmarks.

Deep zooms stay interesting only if the zoom center sits *exactly* on a
point with detail at every scale. Two families qualify:

* Misiurewicz points (pre-periodic critical orbit): self-similar spirals and
  dendrites at every depth. Root of f^(k+p)(0) - f^k(0) = 0.
* Nuclei of minibrots (periodic critical orbit): the zoom ends on a tiny copy
  of the whole set. Root of f^p(0) = 0.

Both are solved with Newton's method in float64, which is exact enough for the
1e13 zoom limit of the float64 kernels.
"""

from __future__ import annotations


def _orbit_with_derivative(c: complex, n: int) -> tuple[complex, complex]:
    z = 0j
    dz = 0j  # d z / d c
    for _ in range(n):
        dz = 2 * z * dz + 1
        z = z * z + c
    return z, dz


def refine_nucleus(c0: complex, period: int, steps: int = 60) -> complex:
    """Newton-solve f^period(0) = 0 starting near ``c0``."""
    c = c0
    for _ in range(steps):
        z, dz = _orbit_with_derivative(c, period)
        if dz == 0:
            break
        step = z / dz
        c -= step
        if abs(step) < 1e-17:
            break
    return c


def refine_misiurewicz(c0: complex, preperiod: int, period: int, steps: int = 80) -> complex:
    """Newton-solve f^(k+p)(0) = f^k(0) starting near ``c0``."""
    c = c0
    for _ in range(steps):
        zk, dzk = _orbit_with_derivative(c, preperiod)
        z, dz = zk, dzk
        for _ in range(period):
            dz = 2 * z * dz + 1
            z = z * z + c
        g = z - zk
        dg = dz - dzk
        if dg == 0:
            break
        step = g / dg
        c -= step
        if abs(step) < 1e-17:
            break
    return c
