"""Deterministic ambient soundtrack: evolving pad chords, soft FM bells, reverb.

Everything is generated with numpy from a seed, so the same arguments always
produce the same samples. Output is float32 stereo at 48 kHz.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

SR = 48_000
PEAK = 0.7  # ~ -3 dBFS master peak

# Chord progressions as semitone offsets from the key root (voiced for a pad).
PROGRESSIONS: list[list[list[int]]] = [
    # i9 - VImaj7 - IIImaj7 - VII6   (Am9, Fmaj7, Cmaj7, G6 in A)
    [[0, 7, 10, 14, 15], [-4, 3, 7, 11, 14], [3, 10, 14, 19, 22], [-2, 5, 9, 12, 14]],
    # Imaj9 - vi9 - IVmaj7 - V sus
    [[0, 4, 7, 11, 14], [-3, 4, 7, 9, 12], [5, 9, 12, 16, 19], [7, 12, 14, 19, 21]],
    # i add9 - iv9 - VI maj7 - v7
    [[0, 3, 7, 14, 15], [5, 8, 12, 15, 19], [-4, 3, 7, 11, 15], [-5, 2, 5, 10, 14]],
]
KEY_ROOTS_HZ = [110.0, 116.54, 123.47, 130.81, 138.59, 146.83]  # A2 .. D3


def _hz(root: float, semitones: float) -> float:
    return root * 2.0 ** (semitones / 12.0)


def _pad_note(t: np.ndarray, freq: float, rng: np.random.Generator) -> np.ndarray:
    """Three detuned soft voices + a quiet octave, spread in stereo. Returns (n, 2)."""
    out = np.zeros((t.size, 2), dtype=np.float64)
    for cents, pan in ((-7.0, 0.15), (0.0, 0.5), (6.0, 0.85)):
        f = freq * 2.0 ** (cents / 1200.0)
        ph = rng.uniform(0, 2 * np.pi)
        v = np.sin(2 * np.pi * f * t + ph) + 0.18 * np.sin(4 * np.pi * f * t + 2 * ph)
        out[:, 0] += v * math.cos(pan * math.pi / 2)
        out[:, 1] += v * math.sin(pan * math.pi / 2)
    return out / 3.0


def _bell(freq: float, dur: float = 4.0) -> np.ndarray:
    """Soft FM bell (ratio 3.5), mono, exponential decay."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 1.6) * (1 - np.exp(-t * 400))
    index = 2.2 * np.exp(-t * 2.5)
    mod = np.sin(2 * np.pi * freq * 3.5 * t)
    return env * np.sin(2 * np.pi * freq * t + index * mod)


def _reverb_ir(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Stereo exponentially-decaying, low-passed noise impulse response."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal((n, 2))
    spec = np.fft.rfft(noise, axis=0)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec *= (1.0 / (1.0 + (freqs / 3500.0) ** 2))[:, None]
    ir = np.fft.irfft(spec, n=n, axis=0) * np.exp(-t * 6.9 / seconds)[:, None]
    return ir / np.sqrt((ir**2).sum(axis=0, keepdims=True))


def _convolve(x: np.ndarray, ir: np.ndarray, block: int = 1 << 17) -> np.ndarray:
    """Per-channel FFT overlap-add convolution; output length len(x) + len(ir) - 1."""
    n, m = x.shape[0], ir.shape[0]
    size = 1 << int(math.ceil(math.log2(block + m - 1)))
    h = np.fft.rfft(ir, n=size, axis=0)
    out = np.zeros((n + m - 1, 2), dtype=np.float64)
    for start in range(0, n, block):
        seg = x[start : start + block]
        y = np.fft.irfft(np.fft.rfft(seg, n=size, axis=0) * h, n=size, axis=0)
        end = min(start + size, out.shape[0])
        out[start:end] += y[: end - start]
    return out


def synthesize(
    duration: float,
    seed: int = 0,
    bell_times: list[float] | None = None,
    loop: bool = False,
    chord_seconds: float = 8.0,
) -> np.ndarray:
    """Return float32 stereo samples of shape (round(duration*SR), 2).

    ``bell_times`` are seconds where a soft bell rings (e.g. zoom milestones).
    With ``loop=True`` chord changes divide the duration evenly and the reverb
    tail wraps to the start, so the audio loops seamlessly.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    root = KEY_ROOTS_HZ[seed % len(KEY_ROOTS_HZ)]
    prog = PROGRESSIONS[(seed // len(KEY_ROOTS_HZ)) % len(PROGRESSIONS)]

    n_chords = max(1, round(duration / chord_seconds))
    seg = duration / n_chords
    fade = min(seg * 0.5, 3.0)

    dry = np.zeros((n, 2), dtype=np.float64)
    # tremolo: slow, whole-number of cycles for loop mode
    trem_cycles = max(1, round(duration / 6.0))
    tt = np.arange(n) / SR
    for i in range(n_chords):
        chord = prog[i % len(prog)]
        t0, t1 = i * seg - fade, (i + 1) * seg + fade
        a, b = max(0, int(t0 * SR)), min(n, int(t1 * SR))
        t = tt[a:b]
        # raised-cosine window over [t0, t1] with plateau => smooth crossfades
        w = np.ones_like(t)
        up = t < t0 + 2 * fade
        down = t > t1 - 2 * fade
        w[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - t0) / (2 * fade))
        w[down] = 0.5 - 0.5 * np.cos(np.pi * (t1 - t[down]) / (2 * fade))
        block = np.zeros((t.size, 2))
        for k, semi in enumerate(chord):
            block += _pad_note(t, _hz(root, semi), rng) * (0.9 if k == 0 else 0.55)
        block += _pad_note(t, _hz(root, chord[0] - 12), rng) * 0.6  # sub
        dry[a:b] += block * w[:, None]
    trem = 0.85 + 0.15 * np.sin(2 * np.pi * trem_cycles * tt / duration)
    dry *= trem[:, None]
    dry *= 0.25 / max(1e-9, np.abs(dry).max())

    bells = np.zeros((n, 2), dtype=np.float64)
    upper = [0, 3, 7, 10, 12, 14, 19, 24]
    for j, bt in enumerate(sorted(bell_times or [])):
        if not 0 <= bt < duration:
            continue
        chord_i = min(n_chords - 1, int(bt / seg))
        tone = prog[chord_i % len(prog)][j % 3] + upper[(j * 3 + seed) % len(upper)] % 12 + 24
        b = _bell(_hz(root, tone))
        s = int(bt * SR)
        pan = 0.3 + 0.4 * ((j * 7 + seed) % 5) / 4
        gains = (math.cos(pan * math.pi / 2), math.sin(pan * math.pi / 2))
        if loop:  # ring-out wraps around to the loop start
            idx = (s + np.arange(b.size)) % n
            for ch in range(2):
                np.add.at(bells[:, ch], idx, b * gains[ch])
        else:
            e = min(n, s + b.size)
            for ch in range(2):
                bells[s:e, ch] += b[: e - s] * gains[ch]
    mix = dry + 0.09 * bells

    wet = _convolve(mix, _reverb_ir(3.0, rng))
    if loop:
        tail = wet[n:]
        wet = wet[:n].copy()
        wet[: tail.shape[0]] += tail[:n]
    else:
        wet = wet[:n]
    out = 0.75 * mix + 0.45 * wet

    if not loop:
        fi, fo = min(n // 4, int(1.5 * SR)), min(n // 4, int(2.5 * SR))
        if fi:
            out[:fi] *= np.linspace(0, 1, fi)[:, None]
        if fo:
            out[n - fo :] *= np.linspace(1, 0, fo)[:, None]
    peak = np.abs(out).max()
    if peak > 0:
        out *= PEAK / peak
    return out.astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> None:
    """Write float stereo samples as 16-bit PCM WAV (stdlib only)."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
