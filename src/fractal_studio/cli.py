"""Command line interface: ``fractal <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import palettes
from .scenes import JULIA_PATHS, JuliaScene, ZoomScene
from .studio import RenderOptions, preview_sheet, render_video
from .targets import MAX_ZOOM, TARGETS, get_target


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _common(p: argparse.ArgumentParser, default_duration: float) -> None:
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--size", type=_size, help="shorthand WxH, e.g. 1920x1080 for long-form")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=float, default=default_duration, help="seconds")
    p.add_argument("--seed", type=int, default=0, help="soundtrack key/progression")
    p.add_argument("--palette", choices=palettes.palette_names(), help="colour palette")
    p.add_argument("--density", type=float, help="palette bands per log-iteration unit")
    p.add_argument(
        "--ss",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="supersampling factor per axis (2 = 2x2 anti-aliasing)",
    )
    p.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    p.add_argument("--stem", help="output file stem (default from target/duration)")
    p.add_argument("--music", type=Path, help="use this audio file instead of the synth")
    p.add_argument("--no-sound", action="store_true", help="silent video (no audio track)")
    p.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    p.add_argument("--preset", default="medium", help="x264 preset")
    p.add_argument("--thumb-at", type=float, help="thumbnail position as fraction (0-1)")
    p.add_argument("-q", "--quiet", action="store_true")


def _options(a: argparse.Namespace) -> RenderOptions:
    w, h = a.size if a.size else (a.width, a.height)
    return RenderOptions(
        width=w,
        height=h,
        fps=a.fps,
        duration=a.duration,
        seed=a.seed,
        palette=a.palette,
        density=a.density,
        ss=a.ss,
        out_dir=a.out,
        stem=a.stem,
        music=a.music,
        sound=not a.no_sound,
        crf=a.crf,
        preset=a.preset,
        thumb_at=a.thumb_at,
        quiet=a.quiet,
        direct=getattr(a, "direct", False),
        quality=getattr(a, "quality", 2),
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fractal", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    z = sub.add_parser("zoom", help="deep zoom into a named target (Mandelbrot/Burning Ship)")
    z.add_argument("--target", default="seahorse_valley", choices=list(TARGETS))
    z.add_argument("--zoom", type=float, help=f"final magnification (max {MAX_ZOOM:.0e})")
    z.add_argument("--rotate", type=float, default=4.0, help="rotation, degrees per second")
    z.add_argument("--palette-speed", type=float, default=0.04, help="palette cycles per second")
    z.add_argument(
        "--direct",
        action="store_true",
        help="iterate every frame (slow, exact) instead of resampling keyframes",
    )
    z.add_argument("--quality", type=int, default=2, help="keyframe oversampling (2-3)")
    _common(z, 8.0)

    j = sub.add_parser("julia", help="seamlessly looping Julia set morph")
    j.add_argument(
        "--path",
        choices=list(JULIA_PATHS),
        default="classic",
        help="closed path that c travels along (classic: |c|=0.7885 circle)",
    )
    j.add_argument("--radius", type=float, help="path scale (default depends on --path)")
    j.add_argument("--phase", type=float, default=0.0, help="start position on the path, turns")
    j.add_argument("--span", type=float, default=2.3, help="view extent of the shorter side")
    j.add_argument("--max-iter", type=int, default=600)
    j.add_argument("--rotate-turns", type=int, default=0, help="full turns per loop")
    j.add_argument("--palette-cycles", type=int, default=1, help="palette cycles per loop")
    _common(j, 8.0)

    pv = sub.add_parser("preview", help="contact sheet of a target at increasing depth")
    pv.add_argument("--target", required=True, choices=list(TARGETS))
    pv.add_argument("--palette", choices=palettes.palette_names())
    pv.add_argument("--out", type=Path, default=Path("out"))

    sub.add_parser("list-targets", help="show curated zoom targets")
    sub.add_parser("list-palettes", help="show colour palettes")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if getattr(a, "music", None) is not None and not a.music.is_file():
        ap.error(f"--music: file not found: {a.music}")
    if a.cmd == "list-targets":
        for t in TARGETS.values():
            print(f"{t.name:18s} {t.kind:13s} zoom {t.zoom:7.0e}  palette {t.palette:7s} {t.desc}")
        return 0
    if a.cmd == "list-palettes":
        for name, p in palettes.PALETTES.items():
            print(f"{name:8s} {p['desc']}")
        return 0
    if a.cmd == "preview":
        path = preview_sheet(a.target, a.out / f"preview_{a.target}.png", a.palette)
        print(path)
        return 0
    opts = _options(a)
    if a.cmd == "zoom":
        tg = get_target(a.target)
        scene: ZoomScene | JuliaScene = ZoomScene(
            tg,
            duration=opts.duration,
            zoom=a.zoom or tg.zoom,
            rotate_deg_per_s=a.rotate,
            palette_cycles_per_s=a.palette_speed,
        )
    else:
        scene = JuliaScene(
            duration=opts.duration,
            path=a.path,
            radius=a.radius,
            span=a.span,
            max_iter=a.max_iter,
            rotate_turns=a.rotate_turns,
            palette_cycles=a.palette_cycles,
            phase=a.phase,
        )
    render_video(scene, opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
