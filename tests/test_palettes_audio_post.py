import json
import math
import wave

import numpy as np
import pytest

from fractal_studio import audio, palettes
from fractal_studio.post import build_post, zoom_words
from fractal_studio.scenes import JuliaScene, ZoomScene, ease_progress
from fractal_studio.targets import MAX_ZOOM, TARGETS, get_target


@pytest.mark.parametrize("name", palettes.palette_names())
def test_palette_lut_shape_and_range(name):
    lut = palettes.build_lut(name)
    assert lut.shape == (palettes.LUT_SIZE, 3)
    assert lut.dtype == np.float32
    assert lut.min() >= 0 and lut.max() <= 255
    # cyclic: last entry blends back to the first (no seam)
    assert np.abs(lut[-1] - lut[0]).max() < 6


def test_unknown_palette():
    with pytest.raises(KeyError):
        palettes.build_lut("nope")


def test_colorize_interior_and_shape():
    lut = palettes.build_lut("ultra")
    interior = palettes.interior_color("ultra")
    nu = np.array([[-1.0, 0.0], [10.0, 100.0]])
    rgb = palettes.colorize(nu, lut, interior, 1.0, 0.0, 0.0)
    assert rgb.shape == (2, 2, 3)
    assert np.array_equal(rgb[0, 0], interior)
    pos = palettes.palette_position(10.0, 1.0, 0.0)
    i = int(pos * palettes.LUT_SIZE)
    assert np.allclose(rgb[1, 0], lut[i], atol=2.0)


def test_downsample_box_filter():
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[0, 0] = 255
    out = palettes.downsample(img, 2)
    assert out.shape == (2, 2, 3)
    assert out[0, 0, 0] == pytest.approx(255 / 4)


def test_audio_length_peak_and_determinism():
    a = audio.synthesize(3.0, seed=4, bell_times=[0.5, 2.0])
    b = audio.synthesize(3.0, seed=4, bell_times=[0.5, 2.0])
    assert a.shape == (3 * audio.SR, 2)
    assert a.dtype == np.float32
    assert np.abs(a).max() == pytest.approx(audio.PEAK, rel=1e-3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, audio.synthesize(3.0, seed=5))
    assert np.abs(a[:10]).max() < 1e-3  # fade in


def test_audio_loop_is_seamless():
    a = audio.synthesize(4.0, seed=1, bell_times=[3.5], loop=True)
    # the jump from the last to the first sample is no bigger than a normal step
    step = np.abs(np.diff(a, axis=0)).max()
    assert np.abs(a[0] - a[-1]).max() <= 2 * step


def test_write_wav(tmp_path):
    p = tmp_path / "x.wav"
    audio.write_wav(p, audio.synthesize(1.0))
    with wave.open(str(p)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 48000, 48000)


def test_ease_progress_monotonic_endpoints():
    us = np.linspace(0, 1, 101)
    e = np.array([ease_progress(u) for u in us])
    assert e[0] == 0 and e[-1] == pytest.approx(1)
    assert np.all(np.diff(e) > 0)


def test_zoom_scene_hits_target_and_clamps():
    tg = get_target("seahorse_valley")
    s = ZoomScene(tg, duration=10, zoom=1e12)
    v0, v1 = s.view(0), s.view(10)
    assert v0.zoom == pytest.approx(1) and v1.zoom == pytest.approx(1e12)
    assert (v1.cx, v1.cy) == pytest.approx((tg.cx, tg.cy), abs=1e-20)
    assert v1.max_iter > v0.max_iter
    assert len(s.bell_times()) >= 5
    with pytest.warns(UserWarning):
        assert ZoomScene(tg, duration=10, zoom=1e16).zoom == MAX_ZOOM


def test_julia_scene_loops():
    s = JuliaScene(duration=6)
    a, b = s.view(0), s.view(6)
    assert (a.jr, a.ji) == pytest.approx((b.jr, b.ji), abs=1e-12)
    assert a.palette_offset % 1 == pytest.approx(b.palette_offset % 1)


def test_targets_valid():
    for t in TARGETS.values():
        assert t.kind in ("mandelbrot", "burning_ship")
        assert 1 < t.zoom <= MAX_ZOOM
        assert t.palette in palettes.PALETTES


def test_post_contract():
    for kind, kw in [
        ("mandelbrot", dict(target_title="Seahorse Valley", zoom=1e13)),
        ("julia", {}),
    ]:
        for dur in (8, 600):
            p = build_post(kind, dur, vertical=dur < 60, **kw)
            json.dumps(p)
            assert 0 < len(p["title"]) <= 100
            assert p["hashtags"] and all(not h.startswith("#") for h in p["hashtags"])
            assert set(p["platforms"]) == {"youtube", "tiktok", "instagram", "facebook"}
            assert p["platforms"]["youtube"]["privacy"] == "private"
            # #shorts only on YouTube, and only for vertical Shorts
            assert "shorts" not in p["hashtags"]
            for plat in ("tiktok", "instagram", "facebook"):
                assert "#shorts" not in json.dumps(p["platforms"][plat])
            assert ("shorts" in p["platforms"]["youtube"]["tags"]) == (dur < 60)


def test_zoom_words():
    assert zoom_words(1e13) == "10 Trillion"
    assert zoom_words(3e9) == "3 Billion"
    assert zoom_words(500) == "500"
    assert math.isclose(1e13, 10 * 1e12)
