"""VoxCPM2 worker process. Runs with the python of the separate VoxCPM2 environment, NOT the service venv.

It imports nothing from `app` so it works in that environment. Protocol: JSON lines. The first line written
is {"ready": true, "sample_rate": N} (or {"error": ...}); then for every request line
{"text", "out", "reference"?, "cfg_value"?, "inference_timesteps"?, "seed"?} it writes the audio to `out`
(16-bit WAV) and answers {"ok": true, "samples": N} or {"ok": false, "error": "..."}. EOF on stdin ends it.
"""
import json
import sys


def main() -> None:
    proto = sys.stdout
    sys.stdout = sys.stderr  # the library prints progress; keep the protocol channel clean
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"

    def send(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        import numpy as np
        import soundfile as sf
        import torch
        from voxcpm import VoxCPM

        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device=device)
        sample_rate = int(getattr(model.tts_model, "sample_rate", 48000))
    except Exception as exc:  # noqa: BLE001 - reported to the service, which fails the job
        send({"error": f"{type(exc).__name__}: {exc}"[:500]})
        return
    send({"ready": True, "sample_rate": sample_rate})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("seed") is not None:
                torch.manual_seed(int(req["seed"]))
            kwargs = {"cfg_value": float(req.get("cfg_value", 2.0)),
                      "inference_timesteps": int(req.get("inference_timesteps", 10))}
            if req.get("reference"):
                kwargs["reference_wav_path"] = req["reference"]
            wav = model.generate(text=req["text"], **kwargs)
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            sf.write(req["out"], audio, sample_rate, subtype="PCM_16")
            send({"ok": True, "samples": int(audio.size)})
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the worker
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})


if __name__ == "__main__":
    main()
