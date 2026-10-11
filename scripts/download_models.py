import os
from pathlib import Path
from huggingface_hub import snapshot_download

assert os.environ.get("HF_HOME", "").lower().startswith("d:"), "HF_HOME must point to D:"
jobs = [
    ("ResembleAI/chatterbox", ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                               "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]),
    ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", None),
    ("facebook/mms-tts-ind", None),
]
for repo, patterns in jobs:
    path = Path(snapshot_download(repo_id=repo, allow_patterns=patterns))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"{repo}: {size/1e9:.2f} GB -> {path}")
