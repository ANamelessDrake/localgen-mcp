"""Game voice effects: plans and ffmpeg filter graphs. Standard library only.

Imported by the MCP server (for validation and tests) and by tts/runner.py, which
runs in the text to speech venv and renders the effect with ffmpeg when it is
installed, or with numpy (render_numpy in runner.py) when it is not.

Effects:
  none     the voice as synthesised
  radio    band limited (about 300 to 3200 Hz), compressed, light crackle and hiss
  robotic  constant phase spectrum (a flat, buzzing monotone) with a short metallic echo
  glitch   a voice struggling on low power: stutters on word onsets, dropouts,
           bursts of heavy bit crushing, and light crushing throughout

intensity (0 to 1) scales each effect; 0.5 is the default.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

EFFECTS = ("none", "radio", "robotic", "glitch")
SAMPLE_RATE = 24000


def check(effect: str, intensity: float) -> None:
    if effect not in EFFECTS:
        raise ValueError(f"effect must be one of {EFFECTS}, got {effect!r}")
    if not 0.0 <= float(intensity) <= 1.0:
        raise ValueError("intensity must be between 0 and 1")


@dataclass(frozen=True)
class Piece:
    """One stretch [start, end) of the dry audio (seconds), played in order to build the glitched line.

    kind: "play" (as is), "repeat" (a stutter copy of the next word's onset), "drop" (silence,
    same length) or "crush" (heavy bit crushing)."""

    start: float
    end: float
    kind: str = "play"


def glitch_plan(duration: float, words: list[dict], seed: int, intensity: float = 0.5) -> list[Piece]:
    """Cut the dry line into pieces: stutters at word onsets, then dropouts and crush bursts between them.

    words: [{"text", "start", "end"}] in seconds, from the synthesiser's timestamps. The plan is a pure
    function of its arguments, so a seed reproduces a take exactly."""
    rng = random.Random(seed)
    duration = max(0.0, float(duration))
    if duration <= 0.05:
        return [Piece(0.0, duration)] if duration > 0 else []
    spoken = [w for w in words if str(w.get("text", ""))[:1].isalnum()
              and w.get("start") is not None and w.get("end") is not None
              and 0.0 <= float(w["start"]) < float(w["end"]) <= duration + 0.05
              and float(w["end"]) - float(w["start"]) >= 0.1]
    n_stutter = round(len(spoken) * (0.15 + 0.35 * intensity))
    if spoken and intensity > 0:
        n_stutter = max(1, n_stutter)
    stutter_words = sorted(rng.sample(spoken, min(n_stutter, len(spoken))), key=lambda w: float(w["start"]))
    stutters = []  # (onset, length, repeats)
    for w in stutter_words:
        start = round(float(w["start"]), 4)
        length = round(min(rng.uniform(0.06, 0.14), float(w["end"]) - start), 4)
        repeats = rng.randint(1, 1 + round(2 * intensity))
        stutters.append((start, length, repeats))

    # Dropouts and crush bursts, kept clear of the stutter onsets so each stutter stays audible.
    events = []  # (start, end, kind)
    n_events = round(duration * (0.6 + 1.6 * intensity))
    busy = [(s - 0.05, s + l + 0.05) for s, l, _ in stutters]
    for _ in range(n_events * 4):
        if len(events) >= n_events:
            break
        kind = "drop" if rng.random() < 0.5 else "crush"
        length = rng.uniform(0.04, 0.10 + 0.12 * intensity) if kind == "drop" else rng.uniform(0.08, 0.25)
        start = round(rng.uniform(0.0, max(0.0, duration - length)), 4)
        end = round(start + length, 4)
        if any(start < b and end > a for a, b in busy):
            continue
        events.append((start, end, kind))
        busy.append((start - 0.03, end + 0.03))

    cuts = sorted([(s, "stutter", (l, r)) for s, l, r in stutters] + [(a, k, b) for a, b, k in events],
                  key=lambda c: c[0])
    pieces: list[Piece] = []
    pos = 0.0
    for start, kind, extra in cuts:
        if start < pos:  # overlaps the previous cut; never play audio out of order
            continue
        if start > pos:
            pieces.append(Piece(round(pos, 4), round(start, 4)))
            pos = start
        if kind == "stutter":
            length, repeats = extra
            pieces.extend(Piece(round(start, 4), round(start + length, 4), "repeat") for _ in range(repeats))
        else:
            pieces.append(Piece(round(start, 4), round(extra, 4), kind))
            pos = extra
    end = round(duration, 4)
    if pos < end:
        pieces.append(Piece(round(pos, 4), end))
    elif pieces and pieces[-1].end > end:  # an event ran to the very end
        pieces[-1] = Piece(pieces[-1].start, end, pieces[-1].kind)
    return [p for p in pieces if p.end > p.start]


def glitch_global_bits(intensity: float) -> int:
    """Bit depth of the light crushing applied to the whole glitched line (16 is clean)."""
    return int(round(12 - 5 * intensity))


def _f(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") or "0"


def ffmpeg_filter(effect: str, intensity: float = 0.5, duration: float = 0.0, words: list[dict] | None = None,
                  seed: int = 0) -> str | None:
    """A -filter_complex graph from input [0:a] to output [out], or None for effect "none"."""
    check(effect, intensity)
    i = float(intensity)
    limiter = "alimiter=limit=0.89:level=disabled"
    if effect == "none":
        return None
    if effect == "radio":
        low, high = int(250 + 150 * i), int(3600 - 800 * i)
        noise = 0.002 + 0.012 * i
        return (f"[0:a]highpass=f={low}:poles=2,highpass=f={low},lowpass=f={high}:poles=2,lowpass=f={high},"
                f"acompressor=threshold=0.08:ratio={_f(4 + 6 * i)}:attack=4:release=60:makeup=3,"
                f"acrusher=bits=10:mode=lin:mix={_f(0.15 + 0.35 * i)}[v];"
                f"anoisesrc=d={_f(duration + 0.5)}:c=pink:r={SAMPLE_RATE}:a={_f(noise)}:seed={seed % 2**31},"
                f"highpass=f={low},lowpass=f={high}[n];"
                f"[v][n]amix=inputs=2:duration=first:normalize=0,{limiter}[out]")
    if effect == "robotic":
        wet = 0.5 + 0.5 * i
        return (f"[0:a]asplit=2[dry][r];"
                "[r]afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)':win_size=512:overlap=0.75,"
                f"aecho=0.8:{_f(0.5 + 0.3 * i)}:7|13:0.35|0.25[wet];"
                f"[dry][wet]amix=inputs=2:duration=first:normalize=0:weights='{_f(1 - wet)} {_f(wet)}',"
                f"{limiter}[out]")
    pieces = glitch_plan(duration, words or [], seed, i)
    if not pieces:
        return f"[0:a]{limiter}[out]"
    crush_bits = max(3, int(round(6 - 2 * i)))
    parts = [f"[0:a]asplit={len(pieces)}" + "".join(f"[s{k}]" for k in range(len(pieces)))]
    for k, p in enumerate(pieces):
        chain = f"[s{k}]atrim=start={_f(p.start)}:end={_f(p.end)},asetpts=PTS-STARTPTS"
        if p.kind == "drop":
            chain += ",volume=0"
        elif p.kind == "crush":
            chain += f",acrusher=bits={crush_bits}:samples={int(round(3 + 5 * i))}:mode=lin:mix=1"
        elif p.kind == "repeat":
            chain += ",volume=0.85"
        parts.append(chain + f"[p{k}]")
    labels = "".join(f"[p{k}]" for k in range(len(pieces)))
    parts.append(f"{labels}concat=n={len(pieces)}:v=0:a=1,"
                 f"acrusher=bits={glitch_global_bits(i)}:mode=lin:mix={_f(0.3 + 0.5 * i)},{limiter}[out]")
    return ";".join(parts)


def ffmpeg_command(ffmpeg: str, src: str, dst: str, graph: str) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", src,
            "-filter_complex", graph, "-map", "[out]", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-c:a", "pcm_s16le", dst]


def ogg_command(ffmpeg: str, src: str, dst: str) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", src,
            "-c:a", "libvorbis", "-q:a", "6", dst]
