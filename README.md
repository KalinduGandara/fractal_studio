# fractal_studio

Automated fractal videos for faceless social channels: Mandelbrot and Burning Ship deep zooms
(down to 10 trillion x) and seamlessly looping Julia set morphs, each with a generated ambient
soundtrack and a ready-to-upload export pack (video, thumbnail, title/description/hashtags).

```bash
uv sync
uv run fractal list-targets
uv run fractal zoom --target seahorse_valley --duration 30          # vertical short
uv run fractal zoom --target spiral_galaxy --size 1920x1080 --duration 600   # 10 min long-form
uv run fractal julia --duration 12                                  # perfect loop
```

Every render writes to `out/` (or `--out DIR`):

| file | content |
|---|---|
| `<stem>.mp4` | H.264 yuv420p (+faststart), AAC 48 kHz stereo, default 1080x1920 @ 30 fps |
| `<stem>_thumb.png` | a frame near the end of the zoom (`--thumb-at 0.9`), full size |
| `<stem>_post.json` | `title`, `description`, `hashtags`, per-platform overrides (YouTube defaults to `private`) |
| `<stem>.json` | render metadata: target, zoom, palette, seed, keyframes, `seconds_per_frame` |

## What you can make

**Deep zoom** (`fractal zoom`): an exponential, gently eased zoom with slow rotation into one of
the curated targets. Every target is a point with detail at every scale (Newton-refined
Misiurewicz points, minibrot nuclei, published high-precision coordinates), so the zoom never
ends in a flat colour field or a black blob. Iterations grow with depth, colouring is continuous
(no banding) and relative to the frame's escape counts so deep frames stay colourful, 2x2
supersampling removes shimmer.

| target | fractal | ends in |
|---|---|---|
| `seahorse_valley` | Mandelbrot | seahorse spirals -> jewelled minibrot cluster |
| `spiral_galaxy` | Mandelbrot | an endless self-similar spiral |
| `hidden_minibrot` | Mandelbrot | double spirals opening into a mini Mandelbrot |
| `starfish` | Mandelbrot | branching starfish lace |
| `star_dendrite` | Mandelbrot | stars inside stars |
| `lightning` | Mandelbrot | electric dendrite |
| `elephant_valley` | Mandelbrot | elephant trunks curling into a spiral |
| `ship_armada` | Burning Ship | a mini ship and its armada |
| `ship_antenna` | Burning Ship | repeating masts and spires |

**Julia morph** (`fractal julia`): the Julia parameter `c` travels once around a closed path
(`classic` |c| = 0.7885 circle, `cardioid` main-cardioid boundary, `bulb` period-2 bulb), so the
last frame flows into the first. Audio loops too.

**Palettes** (`--palette`): ultra, fire, ocean, aurora, neon, gold, sunset, ice. The palette slowly
cycles over time (`--palette-speed`); `--density` sets how many colour bands appear.

**Sound**: a deterministic ambient pad (chord progression and key chosen by `--seed`) with soft FM
bells at every 10x of zoom (or every 1/8 of a Julia loop) and convolution reverb. Use
`--music track.mp3` to loop/fade your own track instead, or `--no-sound`.

## Speed (Apple M1 Pro, 8 cores)

Zooms use a keyframe renderer: one oversampled keyframe per 2x of zoom (~45 for 1e13), every
video frame is resampled from the two keyframes around its scale and coloured individually. The
keyframe cost is fixed per zoom depth (not per frame), so long videos only add the cheap per-frame step:

| render | wall time | s/frame |
|---|---|---|
| `zoom seahorse_valley`, 8 s, 1080x1920 @ 30, zoom 1e13, 45 keyframes | 9 min 21 s | 2.34 (keyframe-dominated) |
| marginal cost per extra zoom frame (resample + colour, 2x2 AA) | | 0.24 |
| -> estimated 10 min long-form zoom (18000 frames) | ~1.3 h | ~0.27 |
| `julia` classic, 8 s, 1080x1920 @ 30 (direct) | 1 min 38 s | 0.41 |
| `zoom --direct`, one deep seahorse frame at 1e13, 2x2 AA | | ~10-13 |

`--direct` iterates every frame instead (exact but ~10-30x slower on deep frames).
`--quality 3` renders sharper keyframes (2.25x keyframe cost).

## Limits

float64 arithmetic limits zooms to 1e13 (larger `--zoom` values are clamped with a warning);
going deeper needs perturbation theory with arbitrary precision references.

## Development

`uv run pytest`, `uv run ruff check src tests`. See `CLAUDE.md` for architecture and how to add
targets, palettes and fractal types.
