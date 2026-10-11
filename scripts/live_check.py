"""Live end-to-end check against a running service (real models). Usage:
    API_KEY=... python scripts/live_check.py http://127.0.0.1:8781 [engine ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

TEXT = ("Bismillah. Hari ini, 10/10/2026 atau 1447 H, kita membahas adab menuntut ilmu. "
        "Allah SWT berfirman: [AYAT]قُلْ هُوَ اللَّهُ أَحَدٌ[/AYAT] "
        "Rasulullah SAW bersabda tentang keutamaan ilmu (HR. Muslim). "
        "Infak pekan ini Rp 1.250.000 dari 35 santri.")


def run(base: str, engine: str, client: httpx.Client, out: Path) -> dict:
    r = client.post(f"{base}/v1/tts", json={"text": TEXT, "engine": engine, "with_srt": True, "params": {"seed": 7}})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    t0 = time.time()
    while True:
        job = client.get(f"{base}/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        busy = client.get(f"{base}/health").json()["busy"]
        assert busy, "health must report busy while a job runs"
        time.sleep(1)
    if job["status"] == "failed":
        return {"engine": engine, "status": "failed", "error": job["error"]}
    audio = client.get(f"{base}/v1/jobs/{job_id}/audio")
    (out / f"{engine}.wav").write_bytes(audio.content)
    res = job["result"]
    return {"engine": engine, "status": "done", "wall_s": round(time.time() - t0, 1),
            "duration_s": res["duration_s"], "rtf": res["rtf"], "chunks": res["chunks"],
            "ayat_markers": [(m["start_s"], m["end_s"], m["source"]) for m in res["ayat_markers"]],
            "devices": res["devices"], "transcript": res["transcript"]}


def main() -> None:
    base = sys.argv[1]
    engines = sys.argv[2:] or ["chatterbox_ml"]
    out = Path("data/live_check"); out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"X-API-Key": os.environ["API_KEY"]}, timeout=60) as client:
        assert client.get(f"{base}/health", headers={"X-API-Key": "wrong"}).status_code == 401
        print("health:", json.dumps(client.get(f"{base}/health").json()))
        for engine in engines:
            print(json.dumps(run(base, engine, client, out), ensure_ascii=False))
        print("health after:", json.dumps(client.get(f"{base}/health").json()))


if __name__ == "__main__":
    main()
