"""Keyframe zoom renderer: render once per 2x of zoom, resample every frame.

A zoom video is mostly the same picture at slowly changing scale, so instead
of iterating every pixel of every frame we render one oversampled *keyframe*
per factor of 2 in zoom (about 45 keyframes for a 1e13 zoom) and build each
video frame by resampling the smooth-iteration field of the two keyframes
that bracket its scale: the finer keyframe k+1 covers the center, keyframe k
the outer ring. Colouring (palette cycling, reference count) still happens
per frame, so the result is visually identical to direct rendering while the
render cost no longer depends on the video duration - a 10 minute zoom costs
the same as an 8 second one.

Keyframes are square (side = frame diagonal) when the view rotates, so any
rotation angle is covered. Only two keyframes are kept in memory.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import kernels
from .scenes import FrameView, ZoomScene, iterations_for


@njit(cache=True, inline="always")
def _bilinear(img: np.ndarray, x: float, y: float) -> float:
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    if a < 0.0 or b < 0.0 or c < 0.0 or d < 0.0:
        # touching the interior: nearest neighbour keeps the set's edge crisp
        return img[y0 + (1 if fy >= 0.5 else 0), x0 + (1 if fx >= 0.5 else 0)]
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy


@njit(cache=True, parallel=True)
def resample(
    outer: np.ndarray,
    outer_px: float,
    inner: np.ndarray,
    inner_px: float,
    span: float,
    angle: float,
    flip: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Sample a (height, width) nu image of a view centered on the keyframes' center."""
    out = np.empty((height, width), dtype=np.float64)
    px = span / min(width, height)
    ca = math.cos(angle)
    sa = math.sin(angle)
    oh, ow = outer.shape
    ih, iw = inner.shape
    for row in prange(height):
        dy = -(row - height * 0.5 + 0.5) * px * flip
        for col in range(width):
            dx = (col - width * 0.5 + 0.5) * px
            ox = dx * ca - dy * sa  # complex offset from the center
            oy = dx * sa + dy * ca
            x = ox / inner_px + iw * 0.5 - 0.5
            y = -oy / (inner_px * flip) + ih * 0.5 - 0.5
            if 0.0 <= x < iw - 1 and 0.0 <= y < ih - 1:
                out[row, col] = _bilinear(inner, x, y)
                continue
            x = ox / outer_px + ow * 0.5 - 0.5
            y = -oy / (outer_px * flip) + oh * 0.5 - 0.5
            x = min(max(x, 0.0), ow - 1.000001)
            y = min(max(y, 0.0), oh - 1.000001)
            out[row, col] = _bilinear(outer, x, y)
    return out


class KeyframeZoom:
    """Produces per-frame nu images for a ``ZoomScene`` from cached keyframes."""

    def __init__(self, scene: ZoomScene, width: int, height: int, ss: int = 2, quality: int = 2):
        self.scene = scene
        self.width, self.height, self.ss = width, height, ss
        tg = scene.target
        self.kind = view_kind = scene.view(0.0).kind
        self.flip = -1.0 if view_kind == kernels.KIND_BURNING_SHIP else 1.0
        m = min(width, height)
        self.base_px = quality * m  # keyframe pixels across its span
        if scene.rotate_deg_per_s != 0.0:
            side = int(math.ceil(quality * math.hypot(width, height))) + 4
            self.kw = self.kh = side
        else:
            self.kw = quality * width + 4
            self.kh = quality * height + 4
        self.span0 = tg.start_span
        self.max_index = int(math.floor(math.log2(scene.zoom))) + 1
        self.cache: dict[int, np.ndarray] = {}
        self.rendered = 0

    def keyframe_span(self, k: int) -> float:
        return self.span0 / 2.0**k

    def keyframe(self, k: int) -> np.ndarray:
        if k not in self.cache:
            for old in [i for i in self.cache if i < k - 1]:
                del self.cache[old]
            tg = self.scene.target
            span = self.keyframe_span(k)
            # render_nu measures span on the shorter side of the array it renders
            arr_span = span * min(self.kw, self.kh) / self.base_px
            nu = kernels.render_nu(
                self.kind,
                tg.cx,
                tg.cy,
                arr_span,
                0.0,
                self.kw,
                self.kh,
                iterations_for(tg, 2.0**k),
                0.0,
                0.0,
            )
            self.cache[k] = nu.astype(np.float32)
            self.rendered += 1
        return self.cache[k]

    def frame_nu(self, view: FrameView) -> np.ndarray:
        """Supersampled nu image (height*ss, width*ss) for ``view``."""
        k = int(math.floor(math.log2(self.span0 / view.span) + 1e-9))
        k = min(max(k, 0), self.max_index - 1)
        outer = self.keyframe(k)
        inner = self.keyframe(k + 1)
        return resample(
            outer,
            self.keyframe_span(k) / self.base_px,
            inner,
            self.keyframe_span(k + 1) / self.base_px,
            view.span,
            view.angle,
            self.flip,
            self.width * self.ss,
            self.height * self.ss,
        )
