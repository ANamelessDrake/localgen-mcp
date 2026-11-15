"""Text to speech runner: synthesise one line with Kokoro-82M on the CPU, apply a game effect, write WAV (and OGG).

Runs in the text to speech venv (<root>/venvs/localgen-tts), never in the MCP server:

    <tts python> runner.py spec.json

spec.json: {"text", "voices": [absolute .pt paths, averaged when several], "lang": "a" or "b",
"speed", "effect", "intensity", "seed", "model_dir", "out_wav", "dry_wav" (or null), "ogg" (bool),
"ffmpeg" (path or null), "threads"}.
The last stdout line is "TTS_RESULT {json}" (see report()). Weights are read from model_dir only;
the environment sets HF_HUB_OFFLINE=1 and hides the GPU, so nothing is downloaded and VRAM is untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import effects  # same folder (this script's directory is first on sys.path)

MARKER = "TTS_RESULT"
REPO_ID = "hexgrad/Kokoro-82M"


def synthesise(spec: dict):
    import numpy as np
    import torch
    from kokoro import KModel, KPipeline

    torch.set_num_threads(int(spec.get("threads") or 4))
    model_dir = Path(spec["model_dir"])
    model = KModel(repo_id=REPO_ID, config=str(model_dir / "config.json"),
                   model=str(model_dir / "kokoro-v1_0.pth")).to("cpu").eval()
    pipeline = KPipeline(spec["lang"], repo_id=REPO_ID, model=model, device="cpu")
    voice = ",".join(spec["voices"])  # Kokoro averages comma separated voice packs
    chunks, words, offset = [], [], 0.0
    with torch.inference_mode():
        for result in pipeline(spec["text"], voice=voice, speed=float(spec["speed"])):
            if result.audio is None:
                continue
            audio = result.audio.detach().cpu().numpy().astype(np.float32)
            for t in result.tokens or []:
                if t.start_ts is not None and t.end_ts is not None:
                    words.append({"text": t.text, "start": round(offset + t.start_ts, 4),
                                  "end": round(offset + t.end_ts, 4)})
            chunks.append(audio)
            offset += len(audio) / effects.SAMPLE_RATE
    if not chunks:
        raise SystemExit("nothing to say: the text produced no phonemes")
    return np.concatenate(chunks), words


# -- numpy fallback for the effects (used only when ffmpeg is not installed) --------


def _crush(x, bits: int, hold: int = 1, mix: float = 1.0):
    import numpy as np
    y = x
    if hold > 1:
        y = np.repeat(y[::hold], hold)[: len(x)]
    levels = 2 ** (bits - 1)
    y = np.round(y * levels) / levels
    return (1 - mix) * x + mix * y


def _band(x, low: float, high: float):
    import numpy as np
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / effects.SAMPLE_RATE)
    spec[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spec, n=len(x))


def render_numpy(x, effect: str, intensity: float, words: list[dict], seed: int):
    """An approximation of the ffmpeg graphs in effects.py, for machines without ffmpeg."""
    import numpy as np
    sr, i = effects.SAMPLE_RATE, float(intensity)
    rng = np.random.default_rng(seed)
    if effect == "radio":
        low, high = 250 + 150 * i, 3600 - 800 * i
        y = _band(x, low, high)
        y = np.tanh(y * (3 + 6 * i)) / np.tanh(3 + 6 * i)
        y = _crush(y, 10, mix=0.15 + 0.35 * i)
        y = y + _band(rng.standard_normal(len(x)) * (0.002 + 0.012 * i), low, high)
    elif effect == "robotic":
        t = np.arange(len(x)) / sr
        ring = x * np.sin(2 * np.pi * 70 * t)
        wet = np.copy(ring)
        for delay_ms, decay in ((7, 0.35), (13, 0.25)):
            d = int(sr * delay_ms / 1000)
            wet[d:] += decay * ring[:-d]
        mix = 0.5 + 0.5 * i
        y = (1 - mix) * x + mix * wet
    elif effect == "glitch":
        out = []
        bits = max(3, int(round(6 - 2 * i)))
        for p in effects.glitch_plan(len(x) / sr, words, seed, i):
            seg = x[int(p.start * sr): int(p.end * sr)]
            if p.kind == "drop":
                seg = np.zeros_like(seg)
            elif p.kind == "crush":
                seg = _crush(seg, bits, int(round(3 + 5 * i)))
            elif p.kind == "repeat":
                seg = seg * 0.85
            out.append(seg)
        y = np.concatenate(out) if out else x
        y = _crush(y, effects.glitch_global_bits(i), mix=0.3 + 0.5 * i)
    else:
        y = x
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * (0.89 / peak) if peak > 0.89 else y).astype(np.float32)


# -- output ---------------------------------------------------------------------------


def stats(path: str) -> dict:
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    return {"sample_rate": int(sr), "channels": int(data.shape[1]), "samples": int(len(mono)),
            "duration_seconds": round(len(mono) / sr, 3) if sr else 0.0,
            "rms_dbfs": round(20 * np.log10(rms), 1) if rms > 0 else None,
            "peak_dbfs": round(20 * np.log10(peak), 1) if peak > 0 else None}


def match_peak(path: str, target: float) -> None:
    """Bring an effected take to the dry take's peak level (at most -1 dBFS), so effects do not
    change loudness much: the robotic spectrum filter, for one, loses about 12 dB."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    if peak > 0 and target > 0:
        sf.write(path, data * (min(target, 0.89) / peak), sr, subtype="PCM_16")


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    effects.check(spec["effect"], spec["intensity"])
    try:
        os.nice(5)  # yield the CPU to interactive work on a shared machine
    except OSError:
        pass
    t0 = time.time()
    audio, words = synthesise(spec)
    t_synth = time.time() - t0
    import soundfile as sf

    out_wav = spec["out_wav"]
    dry_wav = spec.get("dry_wav")
    ffmpeg = spec.get("ffmpeg")
    effect, intensity, seed = spec["effect"], float(spec["intensity"]), int(spec["seed"])
    duration = len(audio) / effects.SAMPLE_RATE
    renderer = "none"
    if effect == "none":
        sf.write(out_wav, audio, effects.SAMPLE_RATE, subtype="PCM_16")
    else:
        if dry_wav:
            sf.write(dry_wav, audio, effects.SAMPLE_RATE, subtype="PCM_16")
        if ffmpeg:
            src = dry_wav or out_wav + ".dry.wav"
            if not dry_wav:
                sf.write(src, audio, effects.SAMPLE_RATE, subtype="PCM_16")
            graph = effects.ffmpeg_filter(effect, intensity, duration, words, seed)
            try:
                subprocess.run(effects.ffmpeg_command(ffmpeg, src, out_wav, graph), check=True,
                               capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise SystemExit(f"ffmpeg failed: {exc.stderr.strip()[-2000:]}")
            finally:
                if not dry_wav:
                    Path(src).unlink(missing_ok=True)
            renderer = "ffmpeg"
        else:
            sf.write(out_wav, render_numpy(audio, effect, intensity, words, seed), effects.SAMPLE_RATE,
                     subtype="PCM_16")
            renderer = "numpy"
        match_peak(out_wav, float(abs(audio).max()))
    ogg = None
    if spec.get("ogg") and ffmpeg:
        ogg = str(Path(out_wav).with_suffix(".ogg"))
        subprocess.run(effects.ogg_command(ffmpeg, out_wav, ogg), check=True, capture_output=True)
    report = {"wav": out_wav, "ogg": ogg, "dry_wav": dry_wav if effect != "none" else None,
              "effect_renderer": renderer, "words": words, "dry_duration_seconds": round(duration, 3),
              "synthesis_seconds": round(t_synth, 2), **stats(out_wav)}
    print(MARKER + " " + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
