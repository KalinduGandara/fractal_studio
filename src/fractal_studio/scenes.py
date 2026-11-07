"""Scenes: map time -> fractal view parameters for each video mode."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

from . import kernels
from .targets import MAX_ZOOM, Target

# Julia paths and their default scale
JULIA_PATHS = {"classic": 0.7885, "cardioid": 0.985, "bulb": 1.0}
KIND_IDS = {"mandelbrot": kernels.KIND_MANDELBROT, "burning_ship": kernels.KIND_BURNING_SHIP}


@dataclass(frozen=True)
class FrameView:
    kind: int
    cx: float
    cy: float
    span: float  # complex extent of the shorter image side
    angle: float
    max_iter: int
    jr: float = 0.0
    ji: float = 0.0
    palette_offset: float = 0.0
    zoom: float = 1.0


def ease_progress(u: float, amount: float = 0.35) -> float:
    """Monotonic easing of [0, 1]: slightly slower at both ends (amount < 1)."""
    return u - amount * math.sin(2 * math.pi * u) / (2 * math.pi)


def clamp_zoom(zoom: float) -> float:
    if zoom > MAX_ZOOM:
        warnings.warn(
            f"zoom {zoom:.3g} exceeds the float64 limit {MAX_ZOOM:.0e}; clamping", stacklevel=2
        )
        return MAX_ZOOM
    if zoom < 1.0:
        raise ValueError("zoom must be >= 1")
    return zoom


def iterations_for(target: Target, zoom: float) -> int:
    """Iteration budget grows linearly with the number of zoom decades."""
    return int(target.iter_base + target.iter_per_decade * math.log10(max(zoom, 1.0)))


@dataclass
class ZoomScene:
    """Exponential zoom, centered on ``target``, from the overview down to ``zoom``."""

    target: Target
    duration: float
    zoom: float
    rotate_deg_per_s: float = 4.0
    palette_cycles_per_s: float = 0.04
    loop: bool = False
    kind_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.zoom = clamp_zoom(self.zoom)
        self.kind_name = self.target.kind

    def zoom_at(self, t: float) -> float:
        u = min(max(t / self.duration, 0.0), 1.0)
        return self.zoom ** ease_progress(u)

    def view(self, t: float) -> FrameView:
        tg = self.target
        z = self.zoom_at(t)
        return FrameView(
            kind=KIND_IDS[tg.kind],
            cx=tg.cx,
            cy=tg.cy,
            span=tg.start_span / z,
            angle=math.radians(self.rotate_deg_per_s * t),
            max_iter=iterations_for(tg, z),
            palette_offset=self.palette_cycles_per_s * t,
            zoom=z,
        )

    def bell_times(self, min_gap: float = 1.2) -> list[float]:
        """Times where the zoom passes each power of ten (easing inverted by bisection)."""
        times: list[float] = []
        for k in range(1, int(math.log10(self.zoom)) + 1):
            lo, hi = 0.0, self.duration
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if math.log10(self.zoom_at(mid)) < k else (lo, mid)
            t = 0.5 * (lo + hi)
            if t < self.duration - 0.5 and (not times or t - times[-1] >= min_gap):
                times.append(t)
        return times


@dataclass
class JuliaScene:
    """Julia set whose parameter c travels once around a closed path -> seamless loop."""

    duration: float
    path: str = "classic"
    radius: float | None = None  # path scale; None -> the path's default
    span: float = 2.3
    max_iter: int = 600
    rotate_turns: int = 0  # whole turns of rotation per loop (keeps the loop seamless)
    palette_cycles: int = 1  # whole palette cycles per loop
    phase: float = 0.0  # start angle on the path, in turns
    loop: bool = True
    kind_name: str = "julia"

    def __post_init__(self) -> None:
        if self.path not in JULIA_PATHS:
            raise ValueError(f"unknown julia path {self.path!r}; choose from {list(JULIA_PATHS)}")
        if self.radius is None:
            self.radius = JULIA_PATHS[self.path]

    def c_at(self, t: float) -> complex:
        th = 2 * math.pi * (t / self.duration + self.phase)
        e = complex(math.cos(th), math.sin(th))
        r = self.radius
        if self.path == "classic":
            # |c| = 0.7885: the classic animation, spirals and filaments all the way round
            return r * e
        if self.path == "cardioid":
            # boundary of the main cardioid: c = e^{it}/2 - e^{2it}/4 (r < 1 stays inside)
            return r * (e / 2 - e * e / 4)
        # "bulb": circle around the period-2 bulb (center -1, radius 1/4)
        return -1.0 + 0.25 * r * e

    def view(self, t: float) -> FrameView:
        c = self.c_at(t)
        u = t / self.duration
        return FrameView(
            kind=kernels.KIND_JULIA,
            cx=0.0,
            cy=0.0,
            span=self.span,
            angle=2 * math.pi * self.rotate_turns * u,
            max_iter=self.max_iter,
            jr=c.real,
            ji=c.imag,
            palette_offset=self.palette_cycles * u,
        )

    def bell_times(self) -> list[float]:
        return [self.duration * k / 8 for k in range(8)]
