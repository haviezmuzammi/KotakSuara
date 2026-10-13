"""Stand-in for app/engines/voxcpm_worker.py in tests: same JSON-lines protocol, no model.

Text "OOM" answers with an out-of-memory error, "CRASH" exits, "HANG" never answers; anything else writes 10 samples
per character at 1000 Hz. Every request is appended to $FAKE_VOXCPM_LOG when set.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

if os.environ.get("FAKE_VOXCPM_FAIL_LOAD"):
    print(json.dumps({"error": "model missing"}), flush=True)
    sys.exit(0)
print(json.dumps({"ready": True, "sample_rate": 1000}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if os.environ.get("FAKE_VOXCPM_LOG"):
        with open(os.environ["FAKE_VOXCPM_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**req, "device": sys.argv[1]}) + "\n")
    if req["text"].endswith("OOM"):
        print(json.dumps({"ok": False, "error": "CUDA out of memory"}), flush=True)
    elif req["text"].endswith("CRASH"):
        sys.exit(3)
    elif req["text"].endswith("HANG"):
        sys.stdin.readline()
    else:
        sf.write(req["out"], np.full(len(req["text"]) * 10, 0.25, dtype=np.float32), 1000, subtype="PCM_16")
        print(json.dumps({"ok": True}), flush=True)
