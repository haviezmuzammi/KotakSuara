# KotakSuara (yassalam-tts-id)

[![tests](https://github.com/haviezmuzammi/KotakSuara/actions/workflows/ci.yml/badge.svg)](https://github.com/haviezmuzammi/KotakSuara/actions/workflows/ci.yml)

Indonesian TTS + voice-clone microservice for Yassalam kajian/dakwah narration (5–15 min videos).
Runs natively on a Windows GPU PC (tested on an RTX 4060, 8 GB) as a service; an n8n server calls it over the LAN.

Pipeline: text → segmenter (`[AYAT]`/Arabic → silence) → Indonesian normalizer → chunker → TTS engine → WAV
(+ optional Whisper SRT).

## Rules baked into the code

- **Quran verses are never synthesized.** `[AYAT]...[/AYAT]` blocks and any Arabic-script span become silence
  (`AYAT_SILENCE_MS`); the job result lists `ayat_markers` (start/end seconds) so a qari recording can be inserted.
  Unclosed or nested tags are rejected with 422.
- **Voice cloning needs recorded consent.** `POST /v1/profiles` requires `consent_by`, `consent_date`,
  `consent_evidence`; reference audio lives in `data/profiles/` (git-ignored).
- **Licenses are explicit.** Every engine reports `license`, `commercial_ok`, `unofficial`; submitting to a
  non-commercial or unofficial engine returns a warning. Only `commercial_ok: true` output may go into monetized videos.
- The PC is kept awake only while jobs run (`SetThreadExecutionState` on the single GPU worker thread).

## Engines

| name | license | commercial_ok | notes |
|---|---|---|---|
| `chatterbox_ml` | MIT | yes | Chatterbox Multilingual, Indonesian via `ms`; zero-shot clone; Perth watermark |
| `azure_id` | Azure paid API | yes | `id-ID-ArdiNeural` / `id-ID-GadisNeural`; needs key + region |
| `mms_id` | CC-BY-NC-4.0 | **no** | comparison only |
| `edge_id` | unofficial endpoint | **no** | comparison only; disable with `ENABLE_UNOFFICIAL_ENGINES=false` |

OpenVoice v2 was evaluated and dropped (librosa pin conflicts with Chatterbox; its `av==10` dependency does not
build for Python 3.11 on Windows). `app/engines/base.py:VoiceConverter` is ready for a future converter.

### Adding an engine

1. Create `app/engines/<name>.py` with a `TTSEngine` subclass: set `info = EngineInfo(...)` with an honest
   `license` / `commercial_ok` / `unofficial` / `vram_mb` (0 = no GPU) / `supports_clone`, and implement
   `load(device)`, `unload()`, `loaded`, `synthesize(text, voice) -> (float32 mono, sample_rate)`;
   override `available()` if it needs credentials.
2. Register it in `app/engines/registry.py`.
3. Add a test with a mocked backend (see `tests/test_engines.py`).

GPU models go through `ModelManager` (max `MAX_LOADED_MODELS` on the GPU, LRU unload, CPU fallback on OOM).

## API (all endpoints require `X-API-Key`)

| method | path | |
|---|---|---|
| GET | `/health` | `busy`, `running_job`, `queue_length`, `loaded_models`, `lexicon_error` |
| GET | `/v1/engines` | engine info + availability |
| POST | `/v1/tts` | JSON `{text, engine?, voice?, profile_id?, params?, ayat_silence_ms?, crossfade_ms?, with_srt?}` → 202 `{job_id, warnings}` |
| GET | `/v1/jobs/{id}` | status, progress, result (`duration_s`, `rtf`, `ayat_markers`, `transcript`, ...) |
| GET | `/v1/jobs/{id}/audio` | WAV (409 until done) |
| GET | `/v1/jobs/{id}/srt` | SRT (with `with_srt` or transcribe jobs) |
| POST | `/v1/transcribe` | multipart `audio` (+ `language`, default `id`) → 202 `{job_id}` |
| POST/GET/DELETE | `/v1/profiles` | voice profiles (multipart create with consent fields) |

`params` for Chatterbox: `exaggeration`, `cfg_weight`, `temperature`, `repetition_penalty`, `min_p`, `top_p`, `seed`.
Azure: `rate`, `pitch` (SSML prosody). Jobs run one at a time; a job stuck longer than `STUCK_JOB_SECONDS`
restarts the process (the service manager restarts it and the job is re-queued once).

## Lexicon

`config/lexicon.yaml` — symbols, abbreviations (case-sensitive), pronunciations (case-insensitive). Hot-reloaded;
an invalid file is ignored (previous version kept) and reported in `/health.lexicon_error`.

## Setup (Windows, all on D:)

```powershell
$env:UV_PYTHON_INSTALL_DIR='D:\AI\uv-python'; $env:UV_CACHE_DIR='D:\AI\uv-cache'
uv python install 3.11
uv venv .venv --python D:\AI\uv-python\cpython-3.11-windows-x86_64-none\python.exe
uv pip sync req-main.txt --python .venv\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
$env:HF_HOME='D:\AI\huggingface'; .venv\Scripts\python.exe scripts\download_models.py
Copy-Item .env.example .env   # then fill API_KEY (24+ chars) and Azure values
```

Dependencies are pinned in `req-main.txt` (compiled from `req-main.in`). Service environment
(`HF_HOME`, `PKUSEG_HOME=D:\AI\pkuseg`, ...) is set by the Windows service, not user variables.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest --cov=app        # unit/API tests, mock engines, no GPU
powershell -File scripts\live_run.ps1                # real models on 127.0.0.1:8781, temporary key
```
