"""Video pipeline: scene -> frames -> ffmpeg, plus the export pack."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__, audio, kernels, palettes
from .keyframes import KeyframeZoom
from .post import build_post
from .render import Encoder, direct_nu, shade, write_png
from .scenes import JuliaScene, ZoomScene


@dataclass
class RenderOptions:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    duration: float = 8.0
    seed: int = 0
    palette: str | None = None  # None -> scene default
    density: float | None = None  # None -> scene default
    ss: int = 2
    out_dir: Path = Path("out")
    stem: str | None = None
    music: Path | None = None
    sound: bool = True
    crf: int = 20
    preset: str = "medium"
    thumb_at: float | None = None  # fraction of the duration; None -> scene default
    direct: bool = False  # zoom: iterate every frame instead of keyframe resampling
    quality: int = 2  # zoom: keyframe oversampling factor
    quiet: bool = False


def _fmt_time(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Progress:
    def __init__(self, total: int, quiet: bool) -> None:
        self.total, self.quiet = total, quiet
        self.start = time.perf_counter()

    def update(self, done: int) -> None:
        if self.quiet:
            return
        el = time.perf_counter() - self.start
        spf = el / max(done, 1)
        eta = spf * (self.total - done)
        sys.stderr.write(
            f"\r  frame {done}/{self.total} {100 * done / self.total:5.1f}%  "
            f"{spf:.2f} s/frame  elapsed {_fmt_time(el)}  ETA {_fmt_time(eta)}   "
        )
        sys.stderr.flush()

    def finish(self) -> float:
        if not self.quiet:
            sys.stderr.write("\n")
        return time.perf_counter() - self.start


def render_video(scene: ZoomScene | JuliaScene, opts: RenderOptions) -> dict:
    """Render ``scene`` to ``<out>/<stem>.mp4`` plus thumb, post and metadata files.

    Frames are streamed straight into ffmpeg, so memory use does not grow with
    the duration. Returns the metadata dict (also written to ``<stem>.json``).
    """
    if opts.width % 2 or opts.height % 2:
        raise ValueError("width and height must be even (yuv420p)")
    if opts.music is not None and not Path(opts.music).is_file():
        raise FileNotFoundError(f"music file not found: {opts.music}")
    zoom_mode = isinstance(scene, ZoomScene)
    tg = scene.target if zoom_mode else None
    palette = opts.palette or (tg.palette if tg else "neon")
    density = opts.density if opts.density is not None else (tg.density if tg else 2.5)
    lut = palettes.build_lut(palette)
    interior = palettes.interior_color(palette)

    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = opts.stem or (
        f"{tg.name}_{opts.duration:g}s" if tg else f"julia_{scene.path}_{opts.duration:g}s"
    )
    mp4 = out_dir / f"{stem}.mp4"
    n_frames = int(round(opts.duration * opts.fps))
    thumb_frac = opts.thumb_at if opts.thumb_at is not None else (0.9 if zoom_mode else 0.3)
    thumb_index = min(n_frames - 1, int(thumb_frac * n_frames))

    sound_kind = "none"
    with tempfile.TemporaryDirectory() as tmp:
        audio_path: Path | None = None
        if opts.music is not None:
            audio_path, sound_kind = Path(opts.music), "music"
        elif opts.sound:
            samples = audio.synthesize(
                opts.duration, seed=opts.seed, bell_times=scene.bell_times(), loop=scene.loop
            )
            audio_path = Path(tmp) / "soundtrack.wav"
            audio.write_wav(audio_path, samples)
            sound_kind = "synth"

        if not opts.quiet:
            print(
                f"rendering {mp4} ({opts.width}x{opts.height} @ {opts.fps} fps, "
                f"{n_frames} frames, ss={opts.ss}, palette={palette})",
                file=sys.stderr,
            )
        enc = Encoder(
            mp4,
            opts.width,
            opts.height,
            opts.fps,
            audio_path,
            crf=opts.crf,
            preset=opts.preset,
            duration=opts.duration,
            audio_is_music=sound_kind == "music",
        )
        prog = Progress(n_frames, opts.quiet)
        nu_ref: float | None = None
        # time constant (s) of the colour-reference smoothing
        alpha = 1.0 - math.exp(-1.0 / (0.5 * opts.fps))
        kz = (
            KeyframeZoom(scene, opts.width, opts.height, opts.ss, opts.quality)
            if zoom_mode and not opts.direct
            else None
        )
        try:
            for i in range(n_frames):
                view = scene.view(i / opts.fps)
                if kz is not None:
                    nu = kz.frame_nu(view)
                else:
                    nu = direct_nu(view, opts.width, opts.height, opts.ss)
                ref = 0.0
                if zoom_mode:
                    # colour relative to the (smoothed) minimum escape count
                    r = palettes.reference_nu(nu)
                    nu_ref = r if nu_ref is None else nu_ref + alpha * (r - nu_ref)
                    ref = nu_ref
                frame = shade(nu, view, lut, interior, density, opts.ss, ref)
                enc.write(frame)
                if i == thumb_index:
                    write_png(out_dir / f"{stem}_thumb.png", frame)
                prog.update(i + 1)
        finally:
            enc.close()
        seconds = prog.finish()

    spf = seconds / n_frames
    meta: dict = {
        "tool": "fractal_studio",
        "version": __version__,
        "mode": "zoom" if zoom_mode else "julia",
        "fractal": scene.kind_name,
        "video": mp4.name,
        "width": opts.width,
        "height": opts.height,
        "fps": opts.fps,
        "duration": opts.duration,
        "frames": n_frames,
        "seed": opts.seed,
        "palette": palette,
        "density": density,
        "supersample": opts.ss,
        "sound": sound_kind,
        "music": str(opts.music) if opts.music else None,
        "thumb_frame": thumb_index,
        "render_seconds": round(seconds, 2),
        "seconds_per_frame": round(spf, 4),
        "renderer": "direct" if kz is None else "keyframes",
    }
    if kz is not None:
        meta.update(keyframes=kz.rendered, keyframe_size=[kz.kw, kz.kh], quality=opts.quality)
    if zoom_mode:
        meta.update(
            target=tg.name,
            center=[tg.cx, tg.cy],
            zoom=scene.zoom,
            rotate_deg_per_s=scene.rotate_deg_per_s,
            palette_cycles_per_s=scene.palette_cycles_per_s,
        )
    else:
        meta.update(
            path=scene.path,
            radius=scene.radius,
            span=scene.span,
            max_iter=scene.max_iter,
            loop=True,
        )
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")

    post = build_post(
        scene.kind_name,
        opts.duration,
        vertical=opts.height > opts.width,
        target_title=tg.title if tg else None,
        target_desc=tg.desc if tg else None,
        zoom=scene.zoom if zoom_mode else None,
        sound=sound_kind,
        supersample=opts.ss,
    )
    (out_dir / f"{stem}_post.json").write_text(
        json.dumps(post, indent=2, ensure_ascii=False) + "\n"
    )
    if not opts.quiet:
        print(f"done: {mp4}  ({_fmt_time(seconds)}, {spf:.3f} s/frame)", file=sys.stderr)
    return meta


def preview_sheet(
    target_name: str,
    out: Path,
    palette: str | None = None,
    tiles: int = 6,
    tile_w: int = 270,
    tile_h: int = 480,
) -> Path:
    """Contact sheet of a target at log-spaced zoom depths (for vetting targets)."""
    from .targets import get_target

    tg = get_target(target_name)
    pal = palette or tg.palette
    lut, interior = palettes.build_lut(pal), palettes.interior_color(pal)
    scene = ZoomScene(tg, duration=1.0, zoom=tg.zoom, rotate_deg_per_s=0.0)
    imgs = []
    for k in range(tiles):
        z = tg.zoom ** (k / (tiles - 1))
        span = tg.start_span / z
        it = int(tg.iter_base + tg.iter_per_decade * math.log10(z))
        nu = kernels.render_nu(
            scene.view(0).kind, tg.cx, tg.cy, span, 0.0, tile_w * 2, tile_h * 2, it, 0.0, 0.0
        )
        rgb = palettes.colorize(nu, lut, interior, tg.density, 0.0, palettes.reference_nu(nu))
        imgs.append(palettes.to_uint8(palettes.downsample(rgb, 2)))
    sheet = np.concatenate(imgs, axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_png(out, sheet)
    return out
