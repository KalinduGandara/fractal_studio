"""Named cyclic colour palettes and the nu -> RGB colouring step.

A palette is a list of sRGB stops (0-255) spaced evenly around a circle; the
last stop blends back into the first, so colour bands never show a seam.
Stops are interpolated with a smooth (cosine) blend into a lookup table.

Colouring maps a smooth iteration count ``nu`` to a palette position with a
log scale relative to a reference count ``nu_ref`` (the frame's minimum
escape count, smoothed over time):

    pos = density * log(1 + nu - nu_ref) + offset

Subtracting ``nu_ref`` makes every zoom depth look alike: the background is
always near the start of the log curve and structure near the boundary gets
progressively more colour bands, so deep frames are neither flat nor noisy.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

LUT_SIZE = 4096

PALETTES: dict[str, dict] = {
    "ultra": {
        "desc": "Classic deep blue / white / amber (Ultra Fractal style)",
        "stops": [(0, 7, 100), (32, 107, 203), (237, 255, 255), (255, 170, 0), (60, 12, 2)],
        "interior": (0, 0, 0),
    },
    "fire": {
        "desc": "Embers: black, crimson, orange, gold",
        "stops": [
            (10, 0, 0),
            (120, 6, 6),
            (230, 70, 10),
            (255, 190, 40),
            (255, 245, 200),
            (200, 60, 10),
        ],
        "interior": (0, 0, 0),
    },
    "ocean": {
        "desc": "Abyss navy, teal, aqua foam",
        "stops": [
            (2, 8, 30),
            (5, 50, 110),
            (0, 140, 160),
            (130, 230, 220),
            (240, 255, 250),
            (20, 90, 140),
        ],
        "interior": (0, 0, 0),
    },
    "aurora": {
        "desc": "Night sky with green and violet aurora",
        "stops": [
            (4, 6, 24),
            (20, 120, 90),
            (120, 255, 160),
            (60, 40, 140),
            (200, 90, 220),
            (20, 20, 70),
        ],
        "interior": (0, 0, 0),
    },
    "neon": {
        "desc": "Synthwave magenta / cyan on deep purple",
        "stops": [(12, 0, 30), (255, 0, 140), (255, 210, 250), (0, 230, 255), (40, 0, 120)],
        "interior": (0, 0, 0),
    },
    "gold": {
        "desc": "Black and gold luxury",
        "stops": [(8, 6, 4), (110, 70, 15), (240, 190, 70), (255, 245, 210), (150, 95, 20)],
        "interior": (0, 0, 0),
    },
    "sunset": {
        "desc": "Indigo, rose, coral, peach",
        "stops": [
            (20, 10, 50),
            (110, 30, 110),
            (240, 80, 110),
            (255, 170, 110),
            (255, 235, 190),
            (70, 20, 80),
        ],
        "interior": (0, 0, 0),
    },
    "ice": {
        "desc": "Glacier whites and steel blues",
        "stops": [(5, 10, 25), (40, 80, 140), (150, 200, 240), (250, 252, 255), (90, 130, 190)],
        "interior": (0, 0, 0),
    },
}

DEFAULT_PALETTE = "ultra"


def palette_names() -> list[str]:
    return list(PALETTES)


def build_lut(name: str, size: int = LUT_SIZE) -> np.ndarray:
    """Return a (size, 3) float32 sRGB lookup table (0-255) for a cyclic palette."""
    if name not in PALETTES:
        raise KeyError(f"unknown palette {name!r}; choose from {', '.join(PALETTES)}")
    stops = np.asarray(PALETTES[name]["stops"], dtype=np.float64)
    k = len(stops)
    pos = np.arange(size) * k / size
    i0 = np.floor(pos).astype(int) % k
    i1 = (i0 + 1) % k
    f = pos - np.floor(pos)
    f = 0.5 - 0.5 * np.cos(np.pi * f)  # smooth blend, zero slope at stops
    lut = stops[i0] * (1.0 - f)[:, None] + stops[i1] * f[:, None]
    return lut.astype(np.float32)


def interior_color(name: str) -> np.ndarray:
    return np.asarray(PALETTES[name]["interior"], dtype=np.float32)


@njit(cache=True, parallel=True)
def colorize(
    nu: np.ndarray,
    lut: np.ndarray,
    interior: np.ndarray,
    density: float,
    offset: float,
    nu_ref: float = 0.0,
) -> np.ndarray:
    """Map smooth iteration counts to float32 RGB (0-255), shape (h, w, 3)."""
    h, w = nu.shape
    n = lut.shape[0]
    out = np.empty((h, w, 3), dtype=np.float32)
    for r in prange(h):
        for c in range(w):
            v = nu[r, c]
            if v < 0.0:
                out[r, c, 0] = interior[0]
                out[r, c, 1] = interior[1]
                out[r, c, 2] = interior[2]
                continue
            p = density * math.log1p(max(v - nu_ref, 0.0)) + offset
            p = (p - math.floor(p)) * n
            i0 = int(p)
            f = p - i0
            i0 = i0 % n
            i1 = (i0 + 1) % n
            for k in range(3):
                out[r, c, k] = lut[i0, k] * (1.0 - f) + lut[i1, k] * f
    return out


def downsample(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter a (h*f, w*f, 3) float image down by ``factor``."""
    if factor == 1:
        return rgb
    h, w, _ = rgb.shape
    return rgb.reshape(h // factor, factor, w // factor, factor, 3).mean(axis=(1, 3))


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def reference_nu(nu: np.ndarray, percentile: float = 0.5) -> float:
    """Low percentile of the escaped pixels' counts (0 if nothing escaped)."""
    sample = nu[::4, ::4]
    ext = sample[sample >= 0.0]
    return float(np.percentile(ext, percentile)) if ext.size else 0.0


def palette_position(nu: float, density: float, offset: float, nu_ref: float = 0.0) -> float:
    """Pure-python reference of the palette position used by ``colorize``."""
    p = density * math.log1p(max(nu - nu_ref, 0.0)) + offset
    return p - math.floor(p)
