# fractal_studio

Python tool that renders fractal videos (Mandelbrot / Burning Ship deep zooms, looping Julia
morphs) with a generated ambient soundtrack and writes the shared **export pack** used by every
video tool in `/Users/kalindu/MyWork/videos` (mp4 + thumb + post json + metadata json). Faceless
social channels (YouTube, Shorts, TikTok, Instagram, Facebook) are the target; monetization is the
goal, so visual quality and "never zoom into a flat/black region" matter more than features.

## Stack
Python 3.12 (uv), numpy, numba (parallel `prange` JIT for the escape-time kernels), stdlib
`wave`/`zlib` (WAV + PNG writing), ffmpeg via subprocess (rawvideo rgb24 pipe -> libx264 + AAC).
Dev: pytest, ruff. No other runtime dependencies without asking the user.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest` - Lint: `uv run ruff check src tests` - Format: `uv run ruff format src tests`
- Zoom short (1080x1920, 30 fps): `uv run fractal zoom --target seahorse_valley --duration 30`
- Long-form (16:9, 10 min): `uv run fractal zoom --target spiral_galaxy --size 1920x1080 --duration 600`
- Julia loop: `uv run fractal julia --duration 12 [--path classic|cardioid|bulb] [--palette neon]`
- Lists: `uv run fractal list-targets`, `uv run fractal list-palettes`
- Vet a target: `uv run fractal preview --target NAME` -> `out/preview_NAME.png` (6 depths)
- Common flags: `--palette --density --seed --fps --ss 1|2|3 --music FILE --no-sound --out DIR --stem NAME --thumb-at 0.9 --crf 20`
- Zoom-only flags: `--zoom 1e12 --rotate DEG_PER_S --palette-speed CYCLES_PER_S --direct --quality 2|3`

## Layout (src/fractal_studio)
- `kernels.py` numba escape-time kernels -> smooth iteration count `nu` (`<0` = interior). `render_nu(kind, ...)` dispatches on `KIND_*`.
- `palettes.py` named cyclic palettes -> LUT; `colorize(nu, lut, interior, density, offset, nu_ref)`; `downsample` (box filter for supersampling).
- `targets.py` curated zoom targets (`Target` dataclass, `TARGETS`, `MAX_ZOOM = 1e13`).
- `refine.py` Newton refinement of Misiurewicz points / minibrot nuclei (how exact targets are made).
- `scenes.py` `ZoomScene` / `JuliaScene`: time -> `FrameView` (center, span, angle, iterations, palette offset) + `bell_times()` for the soundtrack.
- `keyframes.py` `KeyframeZoom`: one oversampled keyframe per 2x zoom, frames resampled from the two bracketing keyframes (render cost independent of duration).
- `render.py` `direct_nu`, `shade`, PNG writer, ffmpeg `Encoder`.
- `audio.py` deterministic pad + FM bells + convolution reverb (48 kHz stereo float32), `write_wav`.
- `post.py` default title/description/hashtags/per-platform copy. `studio.py` the pipeline. `cli.py` argparse.

## Export pack contract (must not break)
`<out>/<stem>.mp4` (H.264 yuv420p +faststart, AAC 48 kHz stereo; default 1080x1920@30),
`<stem>_thumb.png` (a frame of the video, same size), `<stem>_post.json`
(`title` <=100 chars, `description`, `hashtags` without `#`, `platforms.{youtube,tiktok,instagram,facebook}`; YouTube privacy defaults to `private`),
`<stem>.json` metadata (params, seed, timings incl. `seconds_per_frame`). Default out dir `out/` (gitignored).

## Conventions
- Determinism: same args -> same frames and same audio samples. Randomness only from `numpy.random.default_rng(seed)`; no wall-clock except progress/`render_seconds`.
- Streaming: frames go straight into ffmpeg; never accumulate frames (long-form must stay O(1) memory).
- float64 only: zooms are clamped to `MAX_ZOOM` (1e13) with a warning. Deeper needs perturbation theory - don't fake it.
- Colour = `density * log(1 + nu - nu_ref) + offset` with `nu_ref` the frame's low-percentile escape count, EMA-smoothed (0.5 s) in zoom mode, `0` in Julia mode (keeps loops exact).
- Julia videos must loop seamlessly: whole turns of path/rotation/palette per loop, audio `loop=True` (reverb tail and bells wrap to the start).
- Before committing visual changes, render a short sample, extract frames with ffmpeg and look at them.
- Commits: conventional (`feat(zoom): ...`, `fix(audio): ...`), ending with the Co-Authored-By line. Never push.

## Recipes
**Add a zoom target**: a deep zoom needs a center *exactly* on detail at every scale. Find one by
(a) a published high-precision coordinate, or (b) `refine.refine_misiurewicz(guess, k, p)` / `refine_nucleus(guess, p)` -
loop over small `k, p` and keep the root nearest the guess (see `tests/test_kernels.py::test_refined_landmarks_are_exact`).
Add a `Target(...)` in `targets.py` (`start_span` so the first frame shows the whole set; `zoom` <= 1e13; `palette`, `density` ~0.8-1.5),
then `uv run fractal preview --target NAME` and inspect every tile: reject anything that turns flat, noisy or black.
Burning Ship: only real-axis points (real nuclei / real Misiurewicz points of z^2+c are also Burning Ship landmarks) stay clean; off-axis is mostly noise.

**Add a palette**: add an entry to `PALETTES` in `palettes.py` (4-6 sRGB stops, cyclic, include a dark stop for contrast). Tests cover shape/range automatically.

**Add a fractal type**: write `<name>_point(cr, ci, max_iter)` in `kernels.py` (return `_smooth(n, |z|^2)` on escape, `INTERIOR` otherwise, keep the periodicity check), add a `KIND_*` id and a branch in `render_nu`, map it in `scenes.KIND_IDS`, add labels/tags in `post.py`, add known-point tests. Non-quadratic formulas need a different smoothing exponent (`log(d)` instead of `log 2` for z^d).

**Add a video mode**: new scene class in `scenes.py` exposing `view(t) -> FrameView`, `bell_times()`, `loop`, `kind_name`; add a subcommand in `cli.py`; extend `post.build_post`.
