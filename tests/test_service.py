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
    assert "subhanahu wa ta'ala" in spoken and "sepuluh ribu rupiah" in spoken and "Hijriah" in spoken  # Latin fallback
    assert all(voice.language is None for _, voice in fake_engine.calls)
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


def test_profiles_require_consent_and_clone_flow(client, fake_engine, tmp_path, settings):
    wav = tmp_path / "ref.wav"
    sf.write(wav, np.zeros(6000, dtype=np.float32), 1000)

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
    stored = sorted(p.name.split(".", 1)[1] for p in settings.profiles_dir.iterdir())
    assert stored == ["original.wav", "wav"]  # original kept as evidence + normalized reference
    assert client.delete(f"/v1/profiles/{profile['id']}").status_code == 204
    assert list(settings.profiles_dir.iterdir()) == []
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


def _post_profile(client, path, name):
    with path.open("rb") as fh:
        return client.post("/v1/profiles", data=_profile_form(),
                           files={"reference_audio": (name, fh, "application/octet-stream")})


def test_profile_accepts_ogg_opus_voice_note(client, tmp_path, settings):
    voice = tmp_path / "voice.ogg"
    tone = 0.1 * np.sin(np.linspace(0, 2000, 48000 * 8)).astype(np.float32)
    sf.write(voice, tone, 48000, format="OGG", subtype="OPUS")
    r = _post_profile(client, voice, "voice.ogg")
    assert r.status_code == 201, r.text
    ref = next(p for p in settings.profiles_dir.iterdir() if p.name.endswith(".wav") and ".original" not in p.name)
    info = sf.info(ref)
    assert info.channels == 1 and 7.5 < info.duration < 8.5


def test_profile_rejects_bad_or_short_audio(client, tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")
    assert _post_profile(client, junk, "junk.wav").status_code == 422
    short = tmp_path / "short.wav"
    sf.write(short, np.zeros(2000, dtype=np.float32), 1000)
    r = _post_profile(client, short, "short.wav")
    assert r.status_code == 422 and "5-120 s" in r.json()["detail"]


def test_audio_as_mp3(client, fake_engine):
    fake_engine.sr = 16000  # MP3 needs a real sample rate
    job = _wait_job(client, client.post("/v1/tts", json={"text": "Halo semua."}).json()["job_id"])
    r = client.get(f"/v1/jobs/{job['id']}/audio?format=mp3")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 100
    assert client.get(f"/v1/jobs/{job['id']}/audio?format=mp3").status_code == 200  # cached
    assert client.get(f"/v1/jobs/{job['id']}/audio?format=flac").status_code == 422


def test_arabic_capable_engine_speaks_only_whole_arabic_sentences_in_arabic(client, fake_engine):
    from app.engines.base import EngineInfo

    fake_engine.info = EngineInfo(**{**fake_engine.info.__dict__, "extra_languages": ("ar",)})
    text = "Assalamu'alaikum. Bismillah. Nabi SAW bersabda. Kita belajar. Alhamdulillah."
    job = _wait_job(client, client.post("/v1/tts", json={"text": text}).json()["job_id"])
    assert job["status"] == "done" and job["result"]["arabic_sentences"] == 2
    calls = [(t, v.language) for t, v in fake_engine.calls]
    assert calls == [("السَّلَامُ عَلَيْكُمْ. بِسْمِ اللهِ.", "ar"),
                     ("Nabi shallallahu 'alaihi wa sallam bersabda. Kita belajar.", None),
                     ("الْحَمْدُ لِلَّهِ.", "ar")]


def test_arabic_terms_can_be_switched_off_per_request(client, fake_engine):
    from app.engines.base import EngineInfo

    fake_engine.info = EngineInfo(**{**fake_engine.info.__dict__, "extra_languages": ("ar",)})
    body = {"text": "Nabi SAW bersabda.", "arabic_terms": False}
    job = _wait_job(client, client.post("/v1/tts", json=body).json()["job_id"])
    assert job["status"] == "done" and job["result"]["arabic_sentences"] == 0
    assert [(t, v.language) for t, v in fake_engine.calls] == [("Nabi shallallahu 'alaihi wa sallam bersabda.", None)]


def test_trim_silence_and_gap():
    from app.audio import trim_silence

    sr = 1000
    audio = np.concatenate([np.zeros(300), 0.5 * np.ones(200), np.zeros(400)]).astype(np.float32)
    trimmed = trim_silence(audio, sr, keep_ms=20)
    assert 200 <= trimmed.size <= 240 and trimmed.max() == 0.5
    assert trim_silence(np.zeros(100, dtype=np.float32), sr).size == 100  # all silent: unchanged
    assert trim_silence(np.zeros(0, dtype=np.float32), sr).size == 0
    b = AudioBuilder(sample_rate=sr, crossfade_ms=100)
    b.add_speech(np.ones(500), sr)
    b.add_gap(90)
    b.add_speech(np.ones(500), sr, crossfade=False)
    assert b.build().size == 1090


def test_arabic_terms_api(client, service):
    terms = client.get("/v1/lexicon/arabic-terms").json()
    assert len(terms) > 50 and all(t["enabled"] for t in terms)
    r = client.put("/v1/lexicon/arabic-terms/off", json={"arabic_terms_off": ["bismillah", "Wallahu a'lam"]})
    assert r.status_code == 200 and r.json()["off"] == ["Bismillah", "Wallahu a'lam"]
    assert "Bismillah" not in service.lexicon.get().arabic_terms
    assert client.put("/v1/lexicon/arabic-terms/off", json={"arabic_terms_off": []}).json()["off"] == []


def test_pronunciation_api(client, service):
    r = client.put("/v1/lexicon/pronunciations", json={"term": "Bismillah", "respelling": "Bismilah"})
    assert r.status_code == 200
    assert any(t["respelling"] == "Bismilah" for t in client.get("/v1/lexicon/arabic-terms").json())
    assert client.put("/v1/lexicon/pronunciations", json={"term": "Bismillah", "respelling": None}).status_code == 200
    assert all(t["respelling"] is None for t in client.get("/v1/lexicon/arabic-terms").json())
    assert client.put("/v1/lexicon/pronunciations", json={"term": ""}).status_code == 422


def test_arabic_term_preview(client, fake_engine, service):
    from app.engines.base import EngineInfo

    fake_engine.info = EngineInfo(**{**fake_engine.info.__dict__, "extra_languages": ("ar",)})
    service.lexicon.set_arabic_off(["Bismillah"])  # even a term switched to Latin can be previewed in Arabic
    r = client.post("/v1/tts", json={"text": "Bismillah.", "preview_arabic_term": "bismillah"})
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert [(t, v.language) for t, v in fake_engine.calls] == [("بِسْمِ اللهِ.", "ar")]
    assert client.post("/v1/tts", json={"text": "x", "preview_arabic_term": "bukan istilah"}).status_code == 422
    assert client.post("/v1/tts", json={"text": "x", "engine": "nc", "preview_arabic_term": "Bismillah"}).status_code == 422


# reading styles ---------------------------------------------------------------
def _write_styles(settings, text):
    import os

    settings.styles_path.write_text(text, encoding="utf-8")
    t = time.time() + 1
    os.utime(settings.styles_path, (t, t))  # same-second edits must still count as a change


def test_styles_listing_and_validation(client):
    styles = {s["name"]: s for s in client.get("/v1/styles").json()}
    assert {"netral", "dokumenter", "sedih", "bahagia", "horor", "liputan"} <= set(styles)
    assert styles["sedih"]["params"]["exaggeration"] == 0.3 and styles["sedih"]["gap_ms"] == 600
    assert styles["horor"]["fx"] and styles["netral"]["params"] == {}
    r = client.post("/v1/tts", json={"text": "Halo.", "style": "jazz"})
    assert r.status_code == 422 and "/v1/styles" in r.json()["detail"]
    assert client.get("/health").json()["styles_error"] is None


def test_style_params_gaps_and_request_override(client, fake_engine):
    text = "Kalimat pertama yang cukup panjang. " * 12  # several chunks
    r = client.post("/v1/tts", json={"text": text, "style": "Sedih", "params": {"temperature": 0.5}})
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    voices = [v for _, v in fake_engine.calls]
    assert len(voices) >= 2
    assert voices[0].params == {"exaggeration": 0.3, "cfg_weight": 0.3, "temperature": 0.5}  # request wins
    result = job["result"]
    assert result["style"] == "sedih" and result["fx_applied"] is False
    spoken = sum(len(t) * 10 for t, _ in fake_engine.calls)  # FakeEngine: 10 samples per character
    gaps = (len(voices) - 1) * 600  # 600 ms at 1000 Hz = 600 samples between chunks
    assert round(result["duration_s"] * 1000) == spoken + gaps


def test_style_fx_runs_ffmpeg_and_rescales_markers(client, fake_engine, settings):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    _write_styles(settings, "styles:\n  lambat:\n    label: Lambat\n    fx: atempo=0.5\n")
    r = client.post("/v1/tts", json={"text": "Pembuka. [AYAT]ayat[/AYAT] Penutup.", "style": "lambat"})
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    result = job["result"]
    assert result["fx_applied"] is True
    spoken = sum(len(t) * 10 for t, _ in fake_engine.calls)
    assert abs(result["duration_s"] - (spoken + 1000) / 1000 * 2) < 0.1  # half speed doubles the length
    marker = result["ayat_markers"][0]
    assert abs((marker["end_s"] - marker["start_s"]) - 2.0) < 0.05  # 1 s ayat pause, stretched with the audio


def test_style_fx_failure_and_missing_ffmpeg(client, settings, monkeypatch):
    import shutil

    _write_styles(settings, "styles:\n  rusak:\n    label: Rusak\n    fx: bukan_filter=1\n")
    if shutil.which("ffmpeg"):
        job = _wait_job(client, client.post("/v1/tts", json={"text": "Halo.", "style": "rusak"}).json()["job_id"])
        assert job["status"] == "failed" and "effect failed" in job["error"]
    monkeypatch.setattr("app.jobs.find_ffmpeg", lambda configured: None)
    job = _wait_job(client, client.post("/v1/tts", json={"text": "Halo.", "style": "rusak"}).json()["job_id"])
    assert job["status"] == "failed" and "FFMPEG_PATH" in job["error"]


def test_styles_file_validation_keeps_last_good(tmp_path):
    from app.styles import Styles, apply_fx, EffectError, find_ffmpeg, parse_styles

    for bad in ("styles: {}", "styles:\n  Jazz: {}", "styles:\n  a1:\n    params: {speed: 1}",
                "styles:\n  a1:\n    params: {temperature: x}", "styles:\n  a1:\n    gap_ms: -1",
                "styles:\n  a1:\n    fx: ''", "styles:\n  a1: [1]", "- 1"):
        with pytest.raises(ValueError):
            parse_styles(bad)
    path = tmp_path / "styles.yaml"
    path.write_text("styles:\n  ok1:\n    label: Ok\n", encoding="utf-8")
    styles = Styles(path)
    assert styles.get(" OK1 ").label == "Ok"
    path.write_text("styles: [", encoding="utf-8")
    import os
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert "ok1" in styles.all() and styles.last_error  # invalid edit rejected, previous kept
    with pytest.raises(FileNotFoundError):
        Styles(tmp_path / "missing.yaml")
    assert find_ffmpeg(str(tmp_path / "nope.exe")) is None
    assert find_ffmpeg(str(path)) == str(path)
    with pytest.raises(EffectError):
        apply_fx(np.zeros(10, dtype=np.float32), 1000, "anull", str(tmp_path / "nope.exe"))
