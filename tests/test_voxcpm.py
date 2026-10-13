import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from app.engines.base import VoiceSpec
from app.engines.model_manager import ModelManager, is_oom
from app.engines.voxcpm2 import VoxCPM2Engine, engine_text
from tests.test_service import _wait_job

FAKE_WORKER = Path(__file__).with_name("fake_voxcpm_worker.py")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VOXCPM_LOG", str(tmp_path / "requests.jsonl"))
    eng = VoxCPM2Engine(sys.executable, hf_home=str(tmp_path / "hf"), tmp_dir=tmp_path / "tmp",
                        load_timeout=30, call_timeout=30, worker=FAKE_WORKER)
    yield eng
    eng.unload()


def _requests(tmp_path):
    return [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]


def test_engine_text_brackets_and_instruction():
    assert engine_text("(HR. Bukhari) Amal tergantung niat.", None) == "HR. Bukhari, Amal tergantung niat."
    assert engine_text("Ini (contoh) saja.", "calm voice") == "(calm voice)Ini, contoh, saja."
    assert engine_text("()", None) == "()"  # nothing left after stripping: keep the original


def test_availability(tmp_path):
    assert VoxCPM2Engine("").available() == (False, "VOXCPM_PYTHON not set")
    ok, reason = VoxCPM2Engine(str(tmp_path / "nope.exe")).available()
    assert not ok and "not found" in reason
    with pytest.raises(RuntimeError, match="not set"):
        VoxCPM2Engine("").load("cuda")


def test_protocol_params_reference_and_cleanup(engine, tmp_path):
    engine.load("cuda")
    assert engine.loaded and engine.sample_rate == 1000
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    spec = VoiceSpec(reference_path=ref, instruction="calm voice",
                     params={"cfg_value": 2.5, "inference_timesteps": 12.0, "exaggeration": 0.4, "seed": 7})
    audio, sr = engine.synthesize("Halo (dunia).", spec)
    assert sr == 1000 and audio.dtype == np.float32 and abs(float(audio[0]) - 0.25) < 0.01
    req = _requests(tmp_path)[0]
    assert req["text"] == "(calm voice)Halo, dunia."
    assert req["reference"] == str(ref) and req["device"] == "cuda"
    assert req["cfg_value"] == 2.5 and req["inference_timesteps"] == 12 and req["seed"] == 7
    assert "exaggeration" not in req  # Chatterbox-only params are not forwarded
    assert list((tmp_path / "tmp").iterdir()) == []  # temp WAV removed
    engine.unload()
    assert not engine.loaded


def test_oom_crash_and_hang(engine, tmp_path):
    engine.load("cuda")
    with pytest.raises(RuntimeError) as exc:
        engine.synthesize("OOM", VoiceSpec())
    assert is_oom(exc.value) and engine.loaded  # worker survives an OOM
    with pytest.raises(RuntimeError, match="exited"):
        engine.synthesize("CRASH", VoiceSpec())
    assert not engine.loaded
    audio, _ = engine.synthesize("lagi", VoiceSpec())  # restarted automatically
    assert audio.size == 40 and engine.loaded
    engine.call_timeout = 0.5
    with pytest.raises(RuntimeError, match="did not answer"):
        engine.synthesize("HANG", VoiceSpec())
    assert not engine.loaded


def test_failed_load(engine, monkeypatch):
    monkeypatch.setenv("FAKE_VOXCPM_FAIL_LOAD", "1")
    with pytest.raises(RuntimeError, match="model missing"):
        engine.load("cuda")
    assert not engine.loaded


def test_manager_retries_on_cpu_after_oom(engine, tmp_path):
    manager = ModelManager(max_loaded=1, device="cuda", empty_cache=lambda: None)
    texts = ["OOM", "pakai cpu"]
    audio, _ = manager.run(engine, lambda e: e.synthesize(texts.pop(0), VoiceSpec()))
    assert audio.size == len("pakai cpu") * 10
    assert [r["device"] for r in _requests(tmp_path)] == ["cuda", "cpu"]
    assert manager.loaded == [{"name": "voxcpm2", "device": "cpu"}]


def test_style_instruction_and_shared_profile_reach_engine(client, fake_engine, service):
    r = client.post("/v1/tts", json={"text": "Halo semua.", "style": "dokumenter"})
    assert _wait_job(client, r.json()["job_id"])["status"] == "done"
    assert fake_engine.calls[-1][1].instruction.startswith("Indonesian narrator, calm")
    r = client.post("/v1/tts", json={"text": "Halo semua."})
    assert _wait_job(client, r.json()["job_id"])["status"] == "done"
    assert fake_engine.calls[-1][1].instruction is None
    # a profile created for one cloning engine is usable by another (consent covers the voice)
    profile_id = service.db.create_profile({"name": "X", "engine": "lain", "reference_path": "x.wav",
                                            "consent_by": "Narator", "consent_date": "2026-10-13",
                                            "consent_evidence": "rekaman", "notes": ""})["id"]
    r = client.post("/v1/tts", json={"text": "Halo.", "profile_id": profile_id})
    assert r.status_code == 202


def test_style_instruction_validation():
    from app.styles import parse_styles

    for bad in ("styles:\n  a1:\n    instruction: ''", "styles:\n  a1:\n    instruction: (calm)",
                "styles:\n  a1:\n    instruction: " + "x" * 301, "styles:\n  a1:\n    instruction: [1]"):
        with pytest.raises(ValueError, match="instruction"):
            parse_styles(bad)
    assert parse_styles("styles:\n  a1:\n    instruction: calm\n")["a1"].as_dict()["instruction"] == "calm"


def _tone(seconds=3.0, sr=16000, amp=0.02):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


def test_normalize_loudness_with_ffmpeg():
    from app.styles import normalize_loudness

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not installed")
    quiet, sr = _tone()
    out, measured = normalize_loudness(quiet, sr, ffmpeg, -16.0)
    assert measured < -30 and abs(len(out) - len(quiet)) < sr * 0.05
    louder = 20 * np.log10(np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(quiet ** 2)))
    assert louder > 10 and float(np.max(np.abs(out))) < 0.9
    silent, measured = normalize_loudness(np.zeros(sr, dtype=np.float32), sr, ffmpeg)
    assert measured is None and not silent.any()


def test_job_loudness_result_and_skips(client, service, fake_engine, monkeypatch):
    processor = service.worker.processor
    processor.settings = dataclasses.replace(processor.settings, loudness_lufs=-16.0)
    fake_engine.synthesize = lambda text, voice: _tone(1.0, 1000, 0.05)
    monkeypatch.setattr("app.jobs.find_ffmpeg", lambda configured: None)
    job = _wait_job(client, client.post("/v1/tts", json={"text": "Halo."}).json()["job_id"])
    assert job["status"] == "done" and job["result"]["loudness"] == {"skipped": "ffmpeg not found"}

    def boom(*a, **k):
        from app.styles import EffectError
        raise EffectError("ffmpeg failed: x")

    monkeypatch.setattr("app.jobs.find_ffmpeg", lambda configured: "ffmpeg")
    monkeypatch.setattr("app.jobs.normalize_loudness", boom)
    job = _wait_job(client, client.post("/v1/tts", json={"text": "Halo."}).json()["job_id"])
    assert job["status"] == "done" and "ffmpeg failed" in job["result"]["loudness"]["skipped"]
    monkeypatch.setattr("app.jobs.normalize_loudness", lambda audio, sr, ff, target: (audio, None))
    job = _wait_job(client, client.post("/v1/tts", json={"text": "Halo."}).json()["job_id"])
    assert job["result"]["loudness"] == {"skipped": "silent audio"}
    monkeypatch.setattr("app.jobs.normalize_loudness", lambda audio, sr, ff, target: (audio[: len(audio) // 2], -30.04))
    r = client.post("/v1/tts", json={"text": "Awal. [AYAT]ayat[/AYAT] Akhir."})
    result = _wait_job(client, r.json()["job_id"])["result"]
    assert result["loudness"] == {"input_lufs": -30.0, "target_lufs": -16.0}
    assert abs(result["ayat_markers"][0]["start_s"] * 2 - 1.0) < 0.05  # markers follow the length change


def test_settings_loudness_parsing():
    from app.config import _loudness

    assert _loudness("off") is None and _loudness("-14") == -14.0
    with pytest.raises(RuntimeError):
        _loudness("-60")


def test_registry_includes_voxcpm2(settings):
    from app.engines.registry import build_engines

    engines = build_engines(dataclasses.replace(settings, voxcpm_python=""))
    assert engines["voxcpm2"].available()[0] is False and engines["voxcpm2"].info.supports_clone
