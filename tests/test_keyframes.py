import numpy as np

from fractal_studio.keyframes import KeyframeZoom
from fractal_studio.render import direct_nu
from fractal_studio.scenes import ZoomScene
from fractal_studio.targets import get_target


def _compare(target: str, t: float, rotate: float) -> tuple[float, float]:
    scene = ZoomScene(get_target(target), duration=10, zoom=1e9, rotate_deg_per_s=rotate)
    kz = KeyframeZoom(scene, 48, 80, ss=1, quality=2)
    view = scene.view(t)
    a = kz.frame_nu(view)
    b = direct_nu(view, 48, 80, 1)
    both = (a >= 0) & (b >= 0)
    rel = np.median(np.abs(a[both] - b[both]) / b[both])
    mismatch = np.mean((a < 0) != (b < 0))
    return rel, mismatch


def test_keyframes_match_direct_rendering():
    for target, t, rot in [
        ("seahorse_valley", 2.5, 4.0),
        ("lightning", 8.0, 4.0),
        ("ship_armada", 5.0, 0.0),
    ]:
        rel, mismatch = _compare(target, t, rot)
        assert rel < 0.02
        assert mismatch < 0.03


def test_keyframe_cache_is_bounded():
    scene = ZoomScene(get_target("lightning"), duration=4, zoom=1e6)
    kz = KeyframeZoom(scene, 16, 16, ss=1)
    for i in range(41):
        kz.frame_nu(scene.view(i / 10))
        assert len(kz.cache) <= 3
    assert kz.rendered <= kz.max_index + 1
