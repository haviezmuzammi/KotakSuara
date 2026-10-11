import threading
import time

import numpy as np
import pytest
import soundfile as sf

from app.audio import AudioBuilder
from app.db import Database
from app.engines.model_manager import ModelManager
from app.power import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, PowerGuard
from app.stt.whisper import Cue, to_srt
from app.worker import JobError, JobWorker
from tests.conftest import API_KEY, FakeEngine, wait_for


# execution state (must be set and cleared on the same worker thread) -------
def test_power_guard_set_and_released_on_worker_thread(tmp_path):
    calls: list[tuple[int, int]] = []
    guard = PowerGuard(setter=lambda flags: calls.append((flags, threading.get_ident())) or 1)
    db = Database(tmp_path / "db.sqlite3")
    seen_threads = []

    def processor(job, progress):
        seen_threads.append(threading.get_ident())
        time.sleep(0.05)
        return {}

    worker = JobWorker(db, processor, guard, idle_wait=0.02)
    worker.start()
    try:
        ids = [db.create_job("tts", {"n": i}) for i in range(2)]
        worker.notify()
        assert wait_for(lambda: all(db.get_job(i)["status"] == "done" for i in ids))
        assert wait_for(lambda: not guard.active)
    finally:
        worker.stop()

    flags = [f for f, _ in calls]
    assert flags == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]  # once per drain, not per job
    threads = {t for _, t in calls}
    assert threads == {worker.thread_id} == set(seen_threads)
    assert worker.thread_id != threading.get_ident()


def test_power_guard_release_from_other_thread_rejected():
    guard = PowerGuard(setter=lambda flags: 1)
    errors = []
    t = threading.Thread(target=guard.keep_awake)
    t.start(); t.join()
    try:
        guard.release()
    except RuntimeError as exc:
        errors.append(exc)
    assert errors


def test_power_guard_idempotent():
    calls = []
    guard = PowerGuard(setter=lambda flags: calls.append(flags) or 0)  # 0 = API failure, logged only
    guard.release()
    guard.keep_awake(); guard.keep_awake()
    guard.release(); guard.release()
    assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


# worker -----------------------------------------------------------------
def test_worker_failures_and_recovery(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    twice = db.create_job("tts", {})
    db.claim_next(); db.recover_interrupted(); db.claim_next()  # interrupted twice
    once = db.create_job("tts", {})
    db.claim_next()  # interrupted once

    assert db.recover_interrupted() == {"requeued": 1, "failed": 1}
    assert db.get_job(twice)["status"] == "failed"
    assert db.get_job(once)["status"] == "queued"

    def processor(job, progress):
        if job["id"] == once:
            raise JobError("expected failure")
        raise ValueError("boom")

    other = db.create_job("tts", {})
    worker = JobWorker(db, processor, PowerGuard(setter=lambda f: 1), idle_wait=0.02)
    worker.start()
    try:
        assert wait_for(lambda: db.get_job(other)["status"] == "failed")
    finally:
        worker.stop()
    assert db.get_job(once)["error"] == "expected failure"
    assert db.get_job(other)["error"] == "ValueError: boom"


def test_watchdog_triggers_on_stuck_job(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    stuck = threading.Event()
    release = threading.Event()

    def processor(job, progress):
        release.wait(5)
        return {}

    worker = JobWorker(db, processor, PowerGuard(setter=lambda f: 1), stuck_seconds=0.3,
                       on_stuck=stuck.set, idle_wait=0.02)
    worker.start()
    try:
        db.create_job("tts", {})
        worker.notify()
        assert stuck.wait(5)
    finally:
        release.set()
        worker.stop()


# model manager ----------------------------------------------------------
def test_model_manager_lru_and_oom_fallback():
    a, b = FakeEngine("a"), FakeEngine("b")
    light = FakeEngine("light", vram_mb=0)
    mm = ModelManager(max_loaded=1, device="cuda", empty_cache=lambda: None)
    mm.run(a, lambda m: None)
    mm.run(b, lambda m: None)
    assert not a.loaded and b.loaded and mm.loaded == [{"name": "b", "device": "cuda"}]
    mm.run(b, lambda m: None)  # already loaded: no reload
    assert mm.ensure(light) == "cpu" and light.loaded

    attempts = []

    def flaky(model):
        attempts.append(model.device)
        if model.device == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return "ok"

    assert mm.run(a, flaky) == "ok"
    assert attempts == ["cuda", "cpu"]
    assert mm.loaded == [{"name": "a", "device": "cpu"}]
    mm.release_fallbacks()
    assert mm.loaded == []

    with pytest.raises(ValueError):
        mm.run(a, lambda m: (_ for _ in ()).throw(ValueError("not oom")))
    mm.ensure(a); mm.ensure(a, device="cpu")
    assert mm.loaded == [{"name": "a", "device": "cpu"}]
    mm.unload_all()
    assert mm.loaded == []


# audio ------------------------------------------------------------------
def test_audio_builder_crossfade_and_markers():
    b = AudioBuilder(sample_rate=1000, crossfade_ms=100)
    b.add_silence(500, "ayat 1", "tag")
    b.add_speech(np.ones(1000), 1000)
    b.add_speech(np.ones(1000) * 2, 1000)  # 100 samples overlap
    b.add_silence(1000, "ayat 2", "arabic")
    b.add_speech(np.zeros(0), 1000)
    audio = b.build()
    assert audio.size == 500 + 1900 + 1000
    assert b.markers[0].start_s == 0.0 and b.markers[0].end_s == 0.5
    assert b.markers[1].start_s == 2.4 and b.markers[1].end_s == 3.4
    assert np.max(np.abs(audio)) <= 1.0
    with pytest.raises(ValueError):
        b.add_speech(np.ones(10), 16000)
    assert AudioBuilder(sample_rate=10).build().size == 0


def test_srt_format():
    srt = to_srt([Cue(0, 1.5, "halo"), Cue(3661.2, 3662, " dunia ")])
    assert srt == "1\n00:00:00,000 --> 00:00:01,500\nhalo\n\n2\n01:01:01,200 --> 01:01:02,000\ndunia\n"


# API --------------------------------------------------------------------
def _wait_job(client, job_id):
    assert wait_for(lambda: client.get(f"/v1/jobs/{job_id}").json()["status"] in {"done", "failed"})
    return client.get(f"/v1/jobs/{job_id}").json()


def test_auth_required(client):
    assert client.get("/health", headers={"X-API-Key": ""}).status_code == 401
    assert client.get("/health", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/docs").status_code == 404
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["busy"] is False and body["queue_length"] == 0


def test_engines_listing(client):
    engines = {e["name"]: e for e in client.get("/v1/engines").json()}
    assert engines["fake"]["available"] is True
    assert engines["off"]["available"] is False and engines["off"]["unavailable_reason"]
    assert engines["nc"]["commercial_ok"] is False


def test_tts_end_to_end_with_ayat_and_srt(client, fake_engine, settings):
    text = ("Bismillah. Allah SWT berfirman: [AYAT]قُلْ هُوَ اللَّهُ أَحَدٌ[/AYAT] "
            "Infak Rp 10.000 pada 1447 H. Doa: رَبَّنَا آتِنَا selesai.")
    r = client.post("/v1/tts", json={"text": text, "with_srt": True, "params": {"seed": 1}})
    assert r.status_code == 202 and r.json()["warnings"] == []
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    result = job["result"]
    spoken = " ".join(t for t, _ in fake_engine.calls)
    assert "قُلْ" not in spoken and "رَبَّنَا" not in spoken  # verses never reach the engine
    assert "subhanahu wa ta'ala" in spoken and "sepuluh ribu rupiah" in spoken and "Hijriah" in spoken
    assert [m["source"] for m in result["ayat_markers"]] == ["tag", "arabic"]
    assert result["ayat_markers"][0]["end_s"] - result["ayat_markers"][0]["start_s"] == 1.0
    assert result["transcript"] == "halo dunia" and result["commercial_ok"] is True

    audio = client.get(f"/v1/jobs/{job['id']}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    data, sr = sf.read(settings.jobs_dir / job["id"] / "audio.wav")
    assert sr == 1000 and abs(len(data) / sr - result["duration_s"]) < 0.01
    assert "00:00:01,000 --> 00:00:02,500" in client.get(f"/v1/jobs/{job['id']}/srt").text


def test_tts_validation(client, service):
    assert client.post("/v1/tts", json={"text": "x", "engine": "nope"}).status_code == 422
    assert client.post("/v1/tts", json={"text": "x", "engine": "off"}).status_code == 422
    assert client.post("/v1/tts", json={"text": "[AYAT] buka"}).status_code == 422
    assert client.post("/v1/tts", json={"text": "[AYAT]x[/AYAT]"}).status_code == 422
    assert client.post("/v1/tts", json={"text": "a" * 40001}).status_code == 422
    assert client.post("/v1/tts", json={"text": "x", "profile_id": "bad"}).status_code == 422
    assert client.post("/v1/tts", json={"text": "x", "profile_id": "0" * 32}).status_code == 404
    assert client.post("/v1/tts", json={"text": "x", "crossfade_ms": 500}).status_code == 422
    r = client.post("/v1/tts", json={"text": "Halo.", "engine": "nc"})
    assert len(r.json()["warnings"]) == 2
    assert client.get("/v1/jobs/unknown").status_code == 404
    assert client.get("/v1/jobs/unknown/audio").status_code == 404


def test_audio_before_done_is_409(client, service):
    service.worker.stop()
    job_id = service.db.create_job("tts", {"engine": "fake", "text": "x"})
    assert client.get(f"/v1/jobs/{job_id}/audio").status_code == 409
    assert client.get("/health").json()["busy"] is True  # queued work counts as busy


def _profile_form(**overrides):
    form = {"name": "Narator A", "consent_by": "Narator A (pemilik suara)", "consent_date": "2026-10-01",
            "consent_evidence": "Formulir persetujuan bertanda tangan, arsip MPZ-2026-01", "engine": "fake"}
    form.update(overrides)
    return {k: v for k, v in form.items() if v is not None}


def test_profiles_require_consent_and_clone_flow(client, fake_engine, tmp_path):
    wav = tmp_path / "ref.wav"
    sf.write(wav, np.zeros(1000, dtype=np.float32), 1000)

    def post(**overrides):
        with wav.open("rb") as fh:
            return client.post("/v1/profiles", data=_profile_form(**overrides),
                               files={"reference_audio": ("ref.wav", fh, "audio/wav")})

    for missing in ("consent_by", "consent_date", "consent_evidence"):
        assert post(**{missing: None}).status_code == 422
    assert post(consent_evidence="ok").status_code == 422
    assert post(consent_date="2999-01-01").status_code == 422
    assert post(engine="nc").status_code == 422
    with wav.open("rb") as fh:
        bad = client.post("/v1/profiles", data=_profile_form(),
                          files={"reference_audio": ("ref.exe", fh, "application/octet-stream")})
    assert bad.status_code == 422

    created = post()
    assert created.status_code == 201
    profile = created.json()
    assert "reference_path" not in profile and profile["consent_by"].startswith("Narator")
    assert [p["id"] for p in client.get("/v1/profiles").json()] == [profile["id"]]

    r = client.post("/v1/tts", json={"text": "Tes suara.", "profile_id": profile["id"]})
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert fake_engine.calls[-1][1].reference_path.exists()

    assert client.post("/v1/tts", json={"text": "x", "engine": "nc", "profile_id": profile["id"]}).status_code == 422
    assert client.delete(f"/v1/profiles/{profile['id']}").status_code == 204
    assert client.delete(f"/v1/profiles/{profile['id']}").status_code == 404
    assert client.get("/v1/profiles").json() == []


def test_transcribe_job(client, tmp_path):
    wav = tmp_path / "in.wav"
    sf.write(wav, np.zeros(1000, dtype=np.float32), 1000)
    with wav.open("rb") as fh:
        r = client.post("/v1/transcribe", files={"audio": ("in.wav", fh, "audio/wav")})
    assert r.status_code == 202
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done" and job["result"]["transcript"] == "halo dunia"
    assert client.get(f"/v1/jobs/{job['id']}/audio").status_code == 404
    assert client.get(f"/v1/jobs/{job['id']}/srt").status_code == 200


def test_job_fails_cleanly_when_engine_errors(client, fake_engine):
    def explode(text, voice):
        raise RuntimeError("engine exploded")

    fake_engine.synthesize = explode
    job = _wait_job(client, client.post("/v1/tts", json={"text": "Halo."}).json()["job_id"])
    assert job["status"] == "failed" and "engine exploded" in job["error"]


def test_busy_flag_while_running(client, fake_engine):
    gate = threading.Event()
    original = fake_engine.synthesize

    def slow(text, voice):
        gate.wait(5)
        return original(text, voice)

    fake_engine.synthesize = slow
    job_id = client.post("/v1/tts", json={"text": "Halo."}).json()["job_id"]
    assert wait_for(lambda: client.get("/health").json()["busy"] is True)
    gate.set()
    _wait_job(client, job_id)
    assert wait_for(lambda: client.get("/health").json()["busy"] is False)


def test_settings_require_strong_key(tmp_path, monkeypatch):
    from app.config import load_settings

    monkeypatch.setenv("API_KEY", "short")
    with pytest.raises(RuntimeError):
        load_settings(tmp_path / "missing.env")
    env = tmp_path / ".env"
    env.write_text("API_KEY=from-file-0123456789abcdefgh\nPORT=9999\nCROSSFADE_MS=900\n", encoding="utf-8")
    monkeypatch.delenv("API_KEY")
    s = load_settings(env)
    assert s.api_key.startswith("from-file") and s.port == 9999 and s.crossfade_ms == 200
    monkeypatch.setenv("API_KEY", API_KEY)
    assert load_settings(env).api_key == API_KEY


def test_client_allowlist(service):
    from fastapi.testclient import TestClient

    from app.service import create_app

    blocked = TestClient(create_app(service), client=("192.168.2.99", 5000), headers={"X-API-Key": API_KEY})
    r = blocked.get("/health")
    assert r.status_code == 403 and r.json() == {"detail": "client not allowed"}


def test_allowed_clients_setting(tmp_path, monkeypatch):
    from app.config import load_settings

    monkeypatch.setenv("API_KEY", API_KEY)
    assert load_settings(tmp_path / "x.env").allowed_clients == ("127.0.0.1", "::1", "192.168.1.60")
    monkeypatch.setenv("ALLOWED_CLIENTS", " 10.0.0.5 , ,127.0.0.1")
    assert load_settings(tmp_path / "x.env").allowed_clients == ("10.0.0.5", "127.0.0.1")
