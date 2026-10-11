import time, os
from pathlib import Path
import torch, torchaudio
out = Path("data/smoke"); out.mkdir(parents=True, exist_ok=True)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
t = time.time()
tts = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print(f"chatterbox load {time.time()-t:.1f}s, vram {torch.cuda.memory_allocated()/1e9:.2f} GB")
text = "Selamat datang di kajian pekan ini. Hari ini kita membahas adab menuntut ilmu."
torch.cuda.reset_peak_memory_stats(); t = time.time()
wav = tts.generate(text, language_id="ms")
dt = time.time() - t; dur = wav.shape[-1] / tts.sr
torchaudio.save(str(out / "chatterbox_ms.wav"), wav.cpu(), tts.sr)
print(f"chatterbox gen {dt:.1f}s for {dur:.1f}s audio (RTF {dt/dur:.2f}), peak vram {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
del tts; torch.cuda.empty_cache()

from faster_whisper import WhisperModel
t = time.time()
m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
segs, info = m.transcribe(str(out / "chatterbox_ms.wav"), language="id")
print(f"whisper ({time.time()-t:.1f}s):", " ".join(s.text.strip() for s in segs))
