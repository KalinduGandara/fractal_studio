"""Curated, named zoom targets.

A deep zoom is only as good as its center: it must sit (to float64 precision)
on a point with structure at *every* scale, otherwise the zoom runs into a
flat colour field or the black interior. All centers below are either
Newton-refined exact landmarks (see ``refine.py``) or published
high-precision coordinates, and each one was checked with
``fractal preview --target NAME`` down to its default ``zoom``.

To add a target: append a ``Target`` below, run ``fractal preview`` on it,
and inspect the contact sheet at every depth before committing.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_ZOOM = 1e13  # float64 limit: beyond this pixels are a few ulps apart


@dataclass(frozen=True)
class Target:
    name: str
    kind: str  # "mandelbrot" | "burning_ship"
    cx: float
    cy: float
    zoom: float  # default final magnification (relative to span 3.0)
    title: str  # human name used in post titles
    desc: str
    palette: str = "ultra"
    density: float = 3.0  # palette cycles per unit of log(1 + nu)
    iter_base: int = 300
    iter_per_decade: int = 250
    start_span: float = 3.0  # view extent (shorter side) at zoom = 1


TARGETS: dict[str, Target] = {}


def _add(t: Target) -> None:
    TARGETS[t.name] = t


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; choose from {', '.join(TARGETS)}")
    return TARGETS[name]


# --- Mandelbrot --------------------------------------------------------------
_add(
    Target(
        "seahorse_valley",
        "mandelbrot",
        -0.7436438870371587,
        0.13182590420531198,
        1e13,
        title="Seahorse Valley",
        desc="Published deep-zoom coordinate in Seahorse Valley; spirals of seahorse tails "
        "all the way down, ending in a tiny jewelled minibrot cluster.",
        palette="ultra",
        density=1.2,
    )
)
_add(
    Target(
        "spiral_galaxy",
        "mandelbrot",
        -0.7756837680090538,
        0.1364673682946901,
        1e13,
        title="the Infinite Spiral",
        desc="Misiurewicz point M(24,1) (Newton-refined): a self-similar spiral that "
        "repeats forever.",
        palette="ocean",
        density=1.2,
    )
)
_add(
    Target(
        "hidden_minibrot",
        "mandelbrot",
        -0.7746806106269039,
        -0.1374168856037867,
        1e13,
        title="a Hidden Mini Mandelbrot",
        desc="Double spirals that open up into a perfect miniature copy of the whole set "
        "at 10 trillion x.",
        palette="sunset",
        density=1.2,
    )
)
_add(
    Target(
        "starfish",
        "mandelbrot",
        0.281717921930775,
        0.5771052841488505,
        1e13,
        title="the Starfish Lace",
        desc="Lacy many-armed starfish structures that keep branching.",
        palette="neon",
        density=1.2,
        start_span=4.8,
    )
)
_add(
    Target(
        "star_dendrite",
        "mandelbrot",
        0.4245127190500396,
        0.2075302281667453,
        1e13,
        title="the Star Dendrite",
        desc="Misiurewicz point M(7,1) (Newton-refined): branching stars inside stars.",
        palette="gold",
        density=1.2,
        start_span=5.0,
    )
)
_add(
    Target(
        "lightning",
        "mandelbrot",
        -0.10109636384562215,
        0.9562865108091416,
        1e13,
        title="the Lightning Tree",
        desc="Misiurewicz point M(4,1) (Newton-refined): an electric three-way dendrite.",
        palette="ice",
        density=0.8,
        start_span=4.0,
    )
)
_add(
    Target(
        "elephant_valley",
        "mandelbrot",
        0.28488453727317437,
        0.011121821889581924,
        1e13,
        title="Elephant Valley",
        desc="Misiurewicz point M(17,1) (Newton-refined) between the elephants: a parade "
        "of trunks curling into an endless spiral.",
        palette="fire",
        density=1.2,
        start_span=4.8,
    )
)

# --- Burning Ship (real-axis landmarks stay clean; off-axis is mostly noise) ---
_add(
    Target(
        "ship_armada",
        "burning_ship",
        -1.7992901541891173,
        0.0,
        1e10,
        title="the Burning Ship Armada",
        desc="Down the antenna of the Burning Ship to a period-21 mini ship surrounded by "
        "its armada.",
        palette="fire",
        density=1.2,
        start_span=4.0,
    )
)
_add(
    Target(
        "ship_antenna",
        "burning_ship",
        -1.5436890126920764,
        0.0,
        1e12,
        title="the Burning Ship Antenna",
        desc="Pre-periodic point M(3,1) on the antenna: masts and spires repeating forever.",
        palette="aurora",
        density=1.2,
        start_span=4.0,
    )
)
