"""Default social copy for the export pack (<stem>_post.json)."""

from __future__ import annotations

import math

BASE_TAGS = [
    "fractal",
    "fractalart",
    "mathart",
    "satisfying",
    "relaxing",
    "trippy",
    "generativeart",
    "ambient",
]
KIND_TAGS = {
    "mandelbrot": ["mandelbrot", "mandelbrotset", "fractalzoom", "infinitezoom", "deepzoom"],
    "burning_ship": ["burningship", "fractalzoom", "infinitezoom", "deepzoom"],
    "julia": ["juliaset", "loop", "perfectloop", "morph", "hypnotic"],
}
KIND_LABEL = {
    "mandelbrot": "Mandelbrot",
    "burning_ship": "Burning Ship Fractal",
    "julia": "Julia Set",
}


def zoom_words(zoom: float) -> str:
    """1e13 -> '10 Trillion', 3e9 -> '3 Billion', 500 -> '500'."""
    exp = int(math.floor(math.log10(zoom) + 1e-9))
    for size, name in ((12, "Trillion"), (9, "Billion"), (6, "Million"), (3, "Thousand")):
        if exp >= size:
            return f"{round(zoom / 10**size):g} {name}"
    return f"{round(zoom):g}"


def _minutes(duration: float) -> str:
    m = round(duration / 60)
    return f"{m} Minute" + ("s" if m != 1 else "")


def _clip(title: str, limit: int = 100) -> str:
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def build_post(
    kind: str,
    duration: float,
    vertical: bool,
    target_title: str | None = None,
    target_desc: str | None = None,
    zoom: float | None = None,
    sound: str = "synth",
    supersample: int = 1,
) -> dict:
    """Return the export-pack post dict (title, description, hashtags, platforms).

    ``sound`` is "synth" (built-in soundtrack), "music" (user file) or "none".
    """
    long_form = duration >= 120
    label = KIND_LABEL[kind]
    if kind == "julia":
        title = (
            "Julia Set Morph - Perfect Fractal Loop"
            if not long_form
            else f"{_minutes(duration)} of Morphing Julia Set Fractals | Ambient Loop "
            "for Sleep & Focus"
        )
        hook = (
            "The Julia set redraws itself as its parameter c travels around the edge of "
            "the Mandelbrot set - every frame is a different fractal, and it loops "
            "perfectly."
        )
    else:
        assert zoom is not None and target_title is not None
        z = zoom_words(zoom)
        title = (
            f"Zooming {z} Times into {target_title} | {label} Deep Zoom"
            if not long_form
            else f"{_minutes(duration)} Infinite Fractal Zoom into {target_title} | "
            "Ambient Music for Sleep & Focus"
        )
        set_name = "Mandelbrot set" if kind == "mandelbrot" else "Burning Ship fractal"
        hook = (
            f"One continuous zoom, {z.lower()} times deeper, into {target_title} of the "
            f"{set_name}. " + (target_desc or "")
        )
    title = _clip(title)

    tags = list(
        dict.fromkeys(
            KIND_TAGS[kind]
            + BASE_TAGS
            + (["sleepmusic", "studymusic"] if long_form else [])
        )
    )
    # #shorts is YouTube-only; on TikTok/Instagram/Facebook it marks a cross-post.
    yt_tags = tags + (["shorts"] if vertical and not long_form else [])
    hashline = " ".join(f"#{t}" for t in tags)
    sound_line = ""
    if sound == "synth":
        sync = (
            "a soft bell rings at every 10x"
            if kind != "julia"
            else "it loops seamlessly with the picture"
        )
        sound_line = f"Soundtrack: generative ambient pad synced to the video ({sync})."
    formula = "z -> (|Re z| + i|Im z|)^2 + c" if kind == "burning_ship" else "z -> z^2 + c"
    aa = f" and {supersample}x{supersample} anti-aliasing" if supersample > 1 else ""
    tech = (
        f"Every frame is computed from the equation {formula} with smooth colouring"
        f"{aa} - no AI imagery."
    )
    cta = "Follow for a new fractal every day. Which fractal should we explore next?"
    description = "\n\n".join(p for p in (hook, tech, sound_line, cta, hashline) if p)
    yt_description = "\n\n".join(
        p for p in (hook, tech, sound_line, cta, " ".join(f"#{t}" for t in yt_tags)) if p
    )

    short_tags = tags[:6] + (["fyp"] if vertical else [])
    caption = f"{title} {' '.join(f'#{t}' for t in short_tags)}"
    return {
        "title": title,
        "description": description,
        "hashtags": tags,
        "platforms": {
            "youtube": {
                "title": title,
                "description": yt_description,
                "tags": yt_tags,
                "privacy": "private",
                "category_id": "24",
                "made_for_kids": False,
            },
            "tiktok": {"caption": caption[:2200]},
            "instagram": {"caption": f"{title}\n\n{cta}\n\n{hashline}"[:2200]},
            "facebook": {"title": title, "description": description},
        },
    }
