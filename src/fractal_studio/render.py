"""Frame rendering, PNG output and the ffmpeg encoder pipe."""

from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

from . import kernels, palettes
from .scenes import FrameView


def direct_nu(view: FrameView, width: int, height: int, ss: int = 1) -> np.ndarray:
    """Iterate every (supersampled) pixel of ``view``: shape (height*ss, width*ss)."""
    return kernels.render_nu(
        view.kind, view.cx, view.cy, view.span, view.angle,
        width * ss, height * ss, view.max_iter, view.jr, view.ji,
    )  # fmt: skip


def shade(
    nu: np.ndarray,
    view: FrameView,
    lut: np.ndarray,
    interior: np.ndarray,
    density: float,
    ss: int = 1,
    nu_ref: float = 0.0,
) -> np.ndarray:
    """Colour a supersampled nu image and box-filter it to uint8 RGB."""
    rgb = palettes.colorize(nu, lut, interior, density, view.palette_offset, nu_ref)
    return palettes.to_uint8(palettes.downsample(rgb, ss))


def render_frame(
    view: FrameView,
    width: int,
    height: int,
    lut: np.ndarray,
    interior: np.ndarray,
    density: float,
    ss: int = 1,
    nu_ref: float = 0.0,
) -> np.ndarray:
    """Render one frame directly as uint8 RGB (height, width, 3)."""
    return shade(direct_nu(view, width, height, ss), view, lut, interior, density, ss, nu_ref)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal dependency-free RGB PNG writer."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


class Encoder:
    """Streams raw RGB frames into ffmpeg -> H.264/AAC MP4 (never holds all frames)."""

    def __init__(
        self,
        out: Path,
        width: int,
        height: int,
        fps: int,
        audio: Path | None,
        crf: int = 20,
        preset: str = "medium",
        duration: float | None = None,
        audio_is_music: bool = False,
    ) -> None:
        cmd = [
            ffmpeg_bin(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if audio is not None:
            if audio_is_music:
                cmd += ["-stream_loop", "-1"]
            cmd += ["-i", str(audio)]
        cmd += [
            "-map",
            "0:v",
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-maxrate",
            "40M",
            "-bufsize",
            "80M",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
        ]
        if audio is not None:
            cmd += ["-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
            if audio_is_music and duration:
                fade = min(3.0, duration / 4)
                cmd += ["-af", f"afade=t=in:d=1,afade=t=out:st={duration - fade:.3f}:d={fade:.3f}"]
            cmd += ["-shortest"]
        cmd += ["-movflags", "+faststart", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
