"""Text to speech (Kokoro-82M), run on the CPU in its own venv by tts/runner.py.

The MCP server imports nothing heavy: it validates the request, writes a small JSON spec,
runs the runner with the text to speech venv's Python (blender_run.run_reporting, the same
pattern as the Blender scripts) and checks the WAV it wrote. ComfyUI and the GPU are never used.

Settings (environment):
  LOCALGEN_TTS_PYTHON     <root>/venvs/localgen-tts/bin/python
  LOCALGEN_TTS_MODEL_DIR  <root>/models/tts/kokoro-82m
  LOCALGEN_TTS_THREADS    4 (CPU threads for synthesis)
  LOCALGEN_TIMEOUT_TTS    600 s
  LOCALGEN_FFMPEG         ffmpeg on PATH (effects and OGG; without it effects use numpy and no OGG is written)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

from . import blender_run, jobs
from .config import Settings
from .tts import effects

RUNNER = Path(__file__).parent / "tts" / "runner.py"
MARKER = "TTS_RESULT"
MAX_TEXT = 2000
SPEED_RANGE = (0.5, 2.0)
DEFAULT_VOICE = "af_heart"
SILENT_DBFS = -60.0

# Kokoro-82M v1.0 English voices with the grades from its VOICES.md (an estimate of training
# data quality and quantity; A is best). Prefix: a = American, b = British; f = female, m = male.
VOICE_GRADES = {
    "af_heart": "A", "af_bella": "A-", "af_nicole": "B-", "af_aoede": "C+", "af_kore": "C+", "af_sarah": "C+",
    "af_alloy": "C", "af_nova": "C", "af_sky": "C-", "af_jessica": "D", "af_river": "D",
    "am_fenrir": "C+", "am_michael": "C+", "am_puck": "C+", "am_echo": "D", "am_eric": "D", "am_liam": "D",
    "am_onyx": "D", "am_santa": "D-", "am_adam": "F+",
    "bf_emma": "B-", "bf_isabella": "C", "bf_alice": "D", "bf_lily": "D",
    "bm_fable": "C", "bm_george": "C", "bm_lewis": "D+", "bm_daniel": "D",
}
ACCENTS = {"a": "American English", "b": "British English"}
GENDERS = {"f": "female", "m": "male"}


class TtsError(RuntimeError):
    pass


@dataclass(frozen=True)
class TtsSettings:
    python: Path
    model_dir: Path
    threads: int
    timeout: int
    ffmpeg: str | None

    @property
    def voices_dir(self) -> Path:
        return self.model_dir / "voices"


def load_settings(settings: Settings) -> TtsSettings:
    def env_path(name: str, default: Path) -> Path:
        raw = os.environ.get(name)
        return Path(raw).expanduser() if raw else default

    ffmpeg = os.environ.get("LOCALGEN_FFMPEG") or shutil.which("ffmpeg")
    return TtsSettings(
        python=env_path("LOCALGEN_TTS_PYTHON", settings.root / "venvs" / "localgen-tts" / "bin" / "python"),
        model_dir=env_path("LOCALGEN_TTS_MODEL_DIR", settings.root / "models" / "tts" / "kokoro-82m"),
        threads=max(1, int(os.environ.get("LOCALGEN_TTS_THREADS") or 4)),
        timeout=int(os.environ.get("LOCALGEN_TIMEOUT_TTS") or 600),
        ffmpeg=ffmpeg if ffmpeg and os.access(ffmpeg, os.X_OK) else None,
    )


def installed_voices(ts: TtsSettings) -> list[str]:
    """English voice packs present on disk (other languages need G2P extras that are not installed)."""
    if not ts.voices_dir.is_dir():
        return []
    return sorted(p.stem for p in ts.voices_dir.glob("*.pt") if re.fullmatch(r"[ab][fm]_[a-z]+", p.stem))


def describe_voice(name: str) -> dict:
    return {"name": name, "accent": ACCENTS.get(name[0], "unknown"), "gender": GENDERS.get(name[1], "unknown"),
            "grade": VOICE_GRADES.get(name)}


def voices_report(settings: Settings) -> dict:
    ts = load_settings(settings)
    names = installed_voices(ts)
    missing = [p for p in (ts.python, ts.model_dir / "config.json", ts.model_dir / "kokoro-v1_0.pth")
               if not p.exists()]
    return {
        "model": "Kokoro-82M v1.0 (Apache 2.0), CPU",
        "installed": not missing and bool(names),
        "missing": [str(p) for p in missing] + ([] if names else [str(ts.voices_dir / "*.pt")]),
        "install_hint": None if not missing and names else "setup/install.sh --with-tts",
        "default_voice": DEFAULT_VOICE,
        "voices": [describe_voice(n) for n in names],
        "effects": list(effects.EFFECTS),
        "ffmpeg": ts.ffmpeg,
        "ogg": bool(ts.ffmpeg),
    }


def parse_voice(voice: str | None, available: list[str]) -> list[str]:
    """"af_heart" or a blend "am_michael,bm_george" (averaged); every name must be installed and
    share one accent (the accent picks the pronunciation rules)."""
    names = [v.strip() for v in (voice or DEFAULT_VOICE).split(",") if v.strip()]
    if not names:
        raise TtsError("voice is empty")
    if len(names) > 4:
        raise TtsError("blend at most 4 voices")
    for n in names:
        if n not in available:
            raise TtsError(f"unknown voice {n!r}; installed: {', '.join(available) or 'none (setup/install.sh --with-tts)'}")
    if len({n[0] for n in names}) > 1:
        raise TtsError("a blend must use one accent: all American (a*) or all British (b*) voices")
    return names


def output_stem(name: str | None, voice_names: list[str], effect: str) -> str:
    if name:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        if not stem:
            raise TtsError("name has no usable characters (letters, digits, _ . -)")
        return stem[:120]
    return f"{'+'.join(voice_names)}_{effect}_{uuid.uuid4().hex[:8]}"


def validate(text: str, speed: float, effect: str, intensity: float) -> str:
    text = (text or "").strip()
    if not text:
        raise TtsError("text is empty")
    if len(text) > MAX_TEXT:
        raise TtsError(f"text is longer than {MAX_TEXT} characters; split it into lines")
    lo, hi = SPEED_RANGE
    if not lo <= float(speed) <= hi:
        raise TtsError(f"speed must be between {lo} and {hi}")
    try:
        effects.check(effect, intensity)
    except ValueError as exc:
        raise TtsError(str(exc)) from None
    return text


def runner_env(settings: Settings, ts: TtsSettings) -> dict[str, str]:
    tmp = settings.root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "LC_ALL", "USER")}
    threads = str(ts.threads)
    return {**keep, "TMPDIR": str(tmp), "HF_HOME": str(settings.root / ".cache" / "huggingface"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
            "PYTHONWARNINGS": "ignore"}


def command(ts: TtsSettings, spec_path: Path) -> list[str]:
    # -E -s: ignore PYTHON* variables and the user site; the runner's own folder stays first on sys.path.
    return [str(ts.python), "-E", "-s", str(RUNNER), str(spec_path)]


def parse_report(stdout: str) -> dict:
    return blender_run.parse_report(stdout, MARKER, "The text to speech runner", TtsError)


def wav_info(path: Path) -> dict:
    """Header facts read with the standard library, independent of the runner's own numbers."""
    with wave.open(str(path), "rb") as w:
        frames, rate = w.getnframes(), w.getframerate()
        return {"sample_rate": rate, "channels": w.getnchannels(), "sample_width_bytes": w.getsampwidth(),
                "duration_seconds": round(frames / rate, 3) if rate else 0.0}


async def speak(settings: Settings, text: str, voice: str | None = None, speed: float = 1.0, effect: str = "none",
                output_dir: str | None = None, intensity: float = 0.5, seed: int | None = None,
                name: str | None = None, keep_dry: bool = True, ogg: bool = True) -> dict:
    text = validate(text, speed, effect, intensity)
    ts = load_settings(settings)
    if not ts.python.is_file():
        raise TtsError(f"text to speech venv not found ({ts.python}); run setup/install.sh --with-tts")
    for f in ("config.json", "kokoro-v1_0.pth"):
        if not (ts.model_dir / f).is_file():
            raise TtsError(f"Kokoro weights missing ({ts.model_dir / f}); run setup/install.sh --with-tts")
    voice_names = parse_voice(voice, installed_voices(ts))
    seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
    outdir = Path(output_dir).expanduser() if output_dir else jobs.default_output_dir(settings, "audio")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(name, voice_names, effect)
    out_wav = (outdir / f"{stem}.wav").resolve()
    dry_wav = out_wav.with_name(f"{stem}_dry.wav") if effect != "none" and keep_dry else None
    work = settings.root / "tmp" / f"tts-{uuid.uuid4().hex[:10]}"
    work.mkdir(parents=True, exist_ok=True)
    spec = {"text": text, "voices": [str(ts.voices_dir / f"{n}.pt") for n in voice_names], "lang": voice_names[0][0],
            "speed": float(speed), "effect": effect, "intensity": float(intensity), "seed": seed,
            "model_dir": str(ts.model_dir), "out_wav": str(out_wav), "dry_wav": str(dry_wav) if dry_wav else None,
            "ogg": bool(ogg and ts.ffmpeg), "ffmpeg": ts.ffmpeg, "threads": ts.threads}
    spec_path = work / "spec.json"
    spec_path.write_text(json.dumps(spec))
    t0 = time.time()
    try:
        report = await blender_run.run_reporting(command(ts, spec_path), MARKER, out_wav, ts.timeout,
                                                 runner_env(settings, ts), "The text to speech runner", TtsError)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    header = wav_info(out_wav)
    warnings = []
    if report.get("rms_dbfs") is None or report["rms_dbfs"] < SILENT_DBFS:
        warnings.append(f"the output is nearly silent (RMS {report.get('rms_dbfs')} dBFS)")
    if not ts.ffmpeg:
        warnings.append("ffmpeg not found: " + ("no OGG written; " if ogg else "")
                        + ("the effect was rendered with the numpy fallback" if effect != "none" else "WAV only"))
    paths = [str(out_wav)] + ([report["ogg"]] if report.get("ogg") else [])
    return {"paths": paths, "wav": str(out_wav), "ogg": report.get("ogg"), "dry_wav": report.get("dry_wav"),
            "text": text, "voice": ",".join(voice_names), "speed": float(speed), "effect": effect,
            "intensity": float(intensity), "seed": seed, "effect_renderer": report.get("effect_renderer"),
            "sample_rate": header["sample_rate"], "channels": header["channels"],
            "duration_seconds": header["duration_seconds"], "rms_dbfs": report.get("rms_dbfs"),
            "peak_dbfs": report.get("peak_dbfs"), "words": report.get("words", []),
            "synthesis_seconds": report.get("synthesis_seconds"), "total_seconds": round(time.time() - t0, 2),
            "warnings": warnings}
