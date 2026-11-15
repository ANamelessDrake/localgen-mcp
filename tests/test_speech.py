"""speak and list_voices: argument validation, the glitch plan, the ffmpeg graphs (run through a real
ffmpeg on a test tone), the runner command and report, and (when the text to speech venv and the
Kokoro weights are installed) real CPU synthesis through speech.speak."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from localgen_mcp import config, speech
from localgen_mcp.tts import effects

REAL_PYTHON = Path("/ai/venvs/localgen-tts/bin/python")
REAL_MODELS = Path("/ai/models/tts/kokoro-82m")
FFMPEG = shutil.which("ffmpeg")

WORDS = [{"text": "Hull", "start": 0.25, "end": 0.45}, {"text": "integrity", "start": 0.45, "end": 1.02},
         {"text": "at", "start": 1.02, "end": 1.19}, {"text": "forty", "start": 1.19, "end": 1.52},
         {"text": "percent", "start": 1.52, "end": 2.1}, {"text": ".", "start": 2.1, "end": 2.21},
         {"text": "Captain", "start": 2.21, "end": 2.7}, {"text": ",", "start": 2.7, "end": 2.74},
         {"text": "I", "start": 2.74, "end": 2.84}, {"text": "am", "start": 2.84, "end": 2.97},
         {"text": "still", "start": 2.97, "end": 3.27}, {"text": "here", "start": 3.27, "end": 4.1}]
DURATION = 4.4


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path))
    models = tmp_path / "models" / "tts" / "kokoro-82m"
    (models / "voices").mkdir(parents=True)
    for name in ("af_heart", "am_michael", "am_fenrir", "bf_emma", "ef_dora"):
        (models / "voices" / f"{name}.pt").write_bytes(b"")
    return config.load()


# -- validation -------------------------------------------------------------------


@pytest.mark.parametrize("text, speed, effect, intensity", [
    ("", 1.0, "none", 0.5), ("   ", 1.0, "none", 0.5), ("x" * 2001, 1.0, "none", 0.5),
    ("hello", 0.4, "none", 0.5), ("hello", 2.5, "none", 0.5),
    ("hello", 1.0, "echo", 0.5), ("hello", 1.0, "glitch", 1.5), ("hello", 1.0, "glitch", -0.1)])
def test_rejects_bad_arguments(text, speed, effect, intensity):
    with pytest.raises(speech.TtsError):
        speech.validate(text, speed, effect, intensity)


def test_accepts_the_limits():
    assert speech.validate("  hello  ", 0.5, "glitch", 1.0) == "hello"
    assert speech.validate("x" * 2000, 2.0, "none", 0.0) == "x" * 2000


def test_voices_are_checked_against_the_installed_english_packs(settings):
    ts = speech.load_settings(settings)
    available = speech.installed_voices(ts)
    assert available == ["af_heart", "am_fenrir", "am_michael", "bf_emma"]  # ef_dora: Spanish, not installed G2P
    assert speech.parse_voice(None, available) == ["af_heart"]
    assert speech.parse_voice(" am_michael , am_fenrir ", available) == ["am_michael", "am_fenrir"]
    with pytest.raises(speech.TtsError, match="unknown voice"):
        speech.parse_voice("am_nobody", available)
    with pytest.raises(speech.TtsError, match="one accent"):
        speech.parse_voice("am_michael,bf_emma", available)
    with pytest.raises(speech.TtsError, match="unknown voice"):
        speech.parse_voice("../../etc/passwd", available)


def test_output_names_are_sanitised():
    assert speech.output_stem("pathfinder/../hull 40%", ["am_michael"], "glitch") == "pathfinder_.._hull_40"
    assert "/" not in speech.output_stem("../../x", ["am_michael"], "none")
    assert speech.output_stem(None, ["am_michael", "am_fenrir"], "radio").startswith("am_michael+am_fenrir_radio_")
    with pytest.raises(speech.TtsError):
        speech.output_stem("///", ["am_michael"], "none")


def test_missing_install_is_reported_not_crashed(settings):
    report = speech.voices_report(settings)
    assert report["installed"] is False and report["install_hint"] == "setup/install.sh --with-tts"
    with pytest.raises(speech.TtsError, match="install.sh --with-tts"):
        asyncio.run(speech.speak(settings, "hello"))


# -- glitch plan -------------------------------------------------------------------


def _timeline(pieces):
    """The dry stretches the plan plays in order, without the stutter copies."""
    return [(p.start, p.end) for p in pieces if p.kind != "repeat"]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("intensity", [0.0, 0.5, 1.0])
def test_glitch_plan_covers_the_line_once_in_order(seed, intensity):
    pieces = effects.glitch_plan(DURATION, WORDS, seed, intensity)
    line = _timeline(pieces)
    assert line[0][0] == 0.0 and line[-1][1] == pytest.approx(DURATION)
    for (a0, a1), (b0, b1) in zip(line, line[1:]):
        assert a1 == pytest.approx(b0)  # no gap, no overlap, never backwards
        assert a0 < a1
    onsets = {w["start"] for w in WORDS}
    for k, p in enumerate(pieces):
        if p.kind == "repeat":
            assert p.start in onsets  # stutters repeat a word's onset ...
            nxt = next(q for q in pieces[k + 1:] if q.kind != "repeat")
            assert nxt.start == p.start  # ... right before the word itself plays


def test_glitch_plan_is_reproducible_and_scales_with_intensity():
    assert effects.glitch_plan(DURATION, WORDS, 3, 0.5) == effects.glitch_plan(DURATION, WORDS, 3, 0.5)
    assert effects.glitch_plan(DURATION, WORDS, 3, 0.5) != effects.glitch_plan(DURATION, WORDS, 4, 0.5)
    count = lambda i, kind: sum(sum(p.kind == kind for p in effects.glitch_plan(DURATION, WORDS, s, i))
                                for s in range(30))
    assert count(1.0, "repeat") > count(0.0, "repeat") > 0
    assert count(1.0, "drop") + count(1.0, "crush") > count(0.0, "drop") + count(0.0, "crush")


def test_glitch_plan_without_word_timings_still_glitches():
    pieces = effects.glitch_plan(3.0, [], 1, 0.8)
    assert not any(p.kind == "repeat" for p in pieces)
    assert any(p.kind in ("drop", "crush") for p in pieces)
    assert effects.glitch_plan(0.0, [], 1, 0.5) == []


# -- ffmpeg graphs -------------------------------------------------------------------


def test_filter_graphs_have_the_expected_stages():
    assert effects.ffmpeg_filter("none") is None
    radio = effects.ffmpeg_filter("radio", 0.5, DURATION, WORDS, 1)
    assert "highpass" in radio and "lowpass" in radio and "anoisesrc" in radio and radio.endswith("[out]")
    assert "afftfilt" in effects.ffmpeg_filter("robotic", 0.5, DURATION, WORDS, 1)
    glitch = effects.ffmpeg_filter("glitch", 0.5, DURATION, WORDS, 1)
    pieces = effects.glitch_plan(DURATION, WORDS, 1, 0.5)
    assert f"asplit={len(pieces)}" in glitch and f"concat=n={len(pieces)}:v=0:a=1" in glitch
    assert glitch.count("volume=0,") + glitch.count("volume=0[") == sum(p.kind == "drop" for p in pieces)
    with pytest.raises(ValueError):
        effects.ffmpeg_filter("reverb")


def _tone(path: Path, seconds: float) -> None:
    sr = effects.SAMPLE_RATE
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * n / sr)))
                               for n in range(int(seconds * sr))))


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("effect", ["radio", "robotic", "glitch"])
def test_filter_graphs_run_in_ffmpeg(tmp_path, effect):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    _tone(src, DURATION)
    graph = effects.ffmpeg_filter(effect, 0.7, DURATION, WORDS, 5)
    subprocess.run(effects.ffmpeg_command(FFMPEG, str(src), str(dst), graph), check=True, capture_output=True)
    with wave.open(str(dst), "rb") as w:
        assert w.getframerate() == effects.SAMPLE_RATE and w.getnchannels() == 1
        seconds = w.getnframes() / w.getframerate()
    if effect == "glitch":  # stutters make the line longer by exactly the repeated stretches
        extra = sum(p.end - p.start for p in effects.glitch_plan(DURATION, WORDS, 5, 0.7) if p.kind == "repeat")
        assert seconds == pytest.approx(DURATION + extra, abs=0.02)
    else:
        assert seconds == pytest.approx(DURATION, abs=0.02)


# -- runner command and report ----------------------------------------------------------


def test_runner_command_and_environment(settings, tmp_path):
    ts = speech.load_settings(settings)
    cmd = speech.command(ts, tmp_path / "spec.json")
    assert cmd[0] == str(settings.root / "venvs" / "localgen-tts" / "bin" / "python")
    assert Path(cmd[-2]) == speech.RUNNER and speech.RUNNER.is_file()
    assert "-I" not in cmd  # the runner imports effects.py from its own folder
    env = speech.runner_env(settings, ts)
    assert env["HF_HUB_OFFLINE"] == "1" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["TMPDIR"] == str(settings.root / "tmp")
    assert not any(k.startswith("PYTHON") and k != "PYTHONWARNINGS" for k in env)


def test_report_is_parsed_from_the_last_marker_line():
    out = 'WARNING noise\nTTS_RESULT {"wav": "a"}\nTTS_RESULT {"wav": "b", "rms_dbfs": -20}\n'
    assert speech.parse_report(out) == {"wav": "b", "rms_dbfs": -20}
    with pytest.raises(speech.TtsError, match="without a report"):
        speech.parse_report("Traceback ...\nTTS_RESULTS {}")


def test_runner_failure_surfaces_its_output(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_TTS_PYTHON", shutil.which("false"))
    ts = speech.load_settings(settings)
    with pytest.raises(speech.TtsError, match="failed"):
        asyncio.run(speech.blender_run.run_reporting([str(ts.python)], speech.MARKER, tmp_path / "x.wav", 10,
                                                     {}, "The text to speech runner", speech.TtsError))


# -- real synthesis (CPU, about 10 to 20 s each) -----------------------------------------


real = pytest.mark.skipif(not (REAL_PYTHON.is_file() and (REAL_MODELS / "kokoro-v1_0.pth").is_file()),
                          reason="text to speech venv or Kokoro weights not installed")


@pytest.fixture
def real_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCALGEN_TTS_PYTHON", str(REAL_PYTHON))
    monkeypatch.setenv("LOCALGEN_TTS_MODEL_DIR", str(REAL_MODELS))
    return config.load()


def _samples(path: Path) -> tuple[int, ...]:
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        return struct.unpack(f"<{n}h", w.readframes(n))


def _rms_dbfs(path: Path) -> float:
    samples = _samples(path)
    return 20 * math.log10(math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768)


def _peak_dbfs(path: Path) -> float:
    return 20 * math.log10(max(abs(s) for s in _samples(path)) / 32768)


@real
@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
def test_real_speech_with_glitch(real_settings, tmp_path):
    # Two lines: Kokoro synthesises each separately, so word timings must carry across chunks.
    out = asyncio.run(speech.speak(real_settings, "Captain,\nI am still here.", "am_michael", 1.0, "glitch",
                                   str(tmp_path / "out"), 0.6, 11, "line"))
    wav = Path(out["wav"])
    assert out["effect_renderer"] == "ffmpeg" and out["sample_rate"] == 24000 and out["channels"] == 1
    assert Path(out["ogg"]).is_file() and Path(out["dry_wav"]).is_file()
    assert 1.0 < out["duration_seconds"] < 6.0
    assert _rms_dbfs(wav) > -45  # independent of the runner's own measurement
    assert [w["text"] for w in out["words"] if w["text"][0].isalpha()] == ["Captain", "I", "am", "still", "here"]
    with wave.open(out["dry_wav"], "rb") as w:
        dry = w.getnframes() / w.getframerate()
    assert out["duration_seconds"] > dry  # the stutters add time
    assert _peak_dbfs(wav) == pytest.approx(min(_peak_dbfs(Path(out["dry_wav"])), -1.0), abs=0.5)
    starts = [w["start"] for w in out["words"]]
    assert starts == sorted(starts) and out["words"][-1]["end"] <= dry + 0.05
    by_text = {w["text"]: w for w in out["words"]}
    assert by_text["I"]["start"] >= by_text["Captain"]["end"]  # the second chunk starts after the first
    assert not list((real_settings.root / "tmp").glob("tts-*"))  # the spec folder is removed


@real
def test_real_speech_numpy_fallback_without_ffmpeg(real_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALGEN_FFMPEG", str(tmp_path / "no-ffmpeg"))
    out = asyncio.run(speech.speak(real_settings, "Power low.", "bf_emma", 1.2, "robotic", str(tmp_path), name="low"))
    assert out["effect_renderer"] == "numpy" and out["ogg"] is None
    assert out["sample_rate"] == 24000 and _rms_dbfs(Path(out["wav"])) > -45
    assert _peak_dbfs(Path(out["wav"])) == pytest.approx(min(_peak_dbfs(Path(out["dry_wav"])), -1.0), abs=0.5)
    assert any("ffmpeg not found" in w for w in out["warnings"])
    assert json.dumps(out)  # the tool result must be JSON serialisable
