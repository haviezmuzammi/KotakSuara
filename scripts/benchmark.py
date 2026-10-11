"""Phase 4 benchmark: same ~300-word script on each engine -> RTF, peak VRAM, peak RAM, rough WER (Whisper)."""
import json, re, sys, time
from dataclasses import replace
from pathlib import Path

import numpy as np, soundfile as sf, torch

sys.path.insert(0, ".")
from app.audio import AudioBuilder
from app.config import REPO_ROOT
from app.engines.base import VoiceSpec
from app.engines.chatterbox_ml import ChatterboxMLEngine
from app.engines.edge_id import EdgeIDEngine
from app.engines.mms_id import MMSIDEngine
from app.jobs import plan_script, render_calls
from app.stt.whisper import WhisperSTT
from app.text.lexicon import Lexicon
from app.text.normalizer_id import normalize, split_language_parts

TEXT = """Bismillah. Segala puji bagi Allah SWT, shalawat dan salam semoga tercurah kepada Nabi Muhammad SAW, keluarga, dan para sahabatnya. Pada hari Ahad, 12/10/2026, atau tahun 1448 H, kita kembali berkumpul dalam majelis ilmu di Pondok Pesantren Markaz Mulazamah Yassalam.
Tema kajian pekan ini adalah adab sebelum ilmu. Para ulama terdahulu, seperti Imam Malik dan Imam Asy-Syafi'i rahimahumallah, sangat menekankan pentingnya adab bagi seorang penuntut ilmu.
Allah SWT berfirman: [AYAT]QS. Al-Mujadilah ayat 11[/AYAT] Artinya, Allah akan mengangkat orang-orang yang beriman di antara kalian dan orang-orang yang diberi ilmu beberapa derajat.
Ayat ini menunjukkan keutamaan ilmu dan orang-orang yang berilmu. Namun ilmu yang bermanfaat harus dihiasi dengan adab: ikhlas dalam niat, menghormati guru, sabar dalam belajar, dan mengamalkan apa yang telah dipelajari. Abu Bakar r.a. dan para sahabat lainnya adalah teladan dalam hal ini.
Program mulazamah tahun ini diikuti oleh 104 santri dari 17 kota. Biaya pendidikan sebagian besar ditanggung oleh para donatur, dengan total kebutuhan sekitar Rp 25.000.000 per bulan. Alhamdulillah, hingga pekan ke-2 bulan ini, 65% kebutuhan tersebut telah terpenuhi. Infak dapat disalurkan melalui MPZ Yassalam Peduli.
Kajian berikutnya insya Allah dimulai pukul 08.30 WIB dan membahas bab ke-3, tentang adab terhadap teman belajar. Silakan membawa kitab dan buku catatan. Semoga Allah memudahkan langkah kita dalam menuntut ilmu. Wallahu a'lam."""

lex_obj = Lexicon(REPO_ROOT / "config" / "lexicon.yaml"); lex = lex_obj.get()
plan = plan_script(TEXT, lex_obj, 250)
words = len(re.findall(r"\w+", re.sub(r"\[AYAT\].*?\[/AYAT\]", "", TEXT)))

def canon(t):
    t = " ".join(fb for _, _, fb in split_language_parts(normalize(t, lex), lex))
    t = re.sub(r"[^\w\s]", " ", t.lower().replace("-", " ").replace("'", ""))
    return t.split()

def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))

ref_words = canon(" ".join(p["text"] for p in plan if p["kind"] == "speech"))
out = Path("data/benchmark"); out.mkdir(parents=True, exist_ok=True)
profiles = {}
import sqlite3
con = sqlite3.connect("data/tts.sqlite3")
row = con.execute("SELECT reference_path FROM profiles WHERE lower(name)='fahmi'").fetchone()
runs = [("chatterbox_ml", ChatterboxMLEngine(), VoiceSpec(params={"seed": 7})),
        ("chatterbox_ml+klon", None, VoiceSpec(reference_path=Path(row[0]) if row else None, params={"seed": 7})),
        ("mms_id", MMSIDEngine(), VoiceSpec(params={"seed": 7})),
        ("edge_id", EdgeIDEngine(), VoiceSpec(voice="id-ID-ArdiNeural"))]
results = []
cb = None
for name, engine, voice in runs:
    if engine is None:
        engine = cb
        if voice.reference_path is None:
            continue
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if not engine.loaded:
        engine.load("cuda")
    load_s = time.time() - t0
    builder = None; synth = 0.0
    for item in plan:
        if item["kind"] == "ayat":
            if builder: builder.add_silence(2000, item["text"], item["source"])
            continue
        for text, lang in render_calls(item["parts"], engine):
            spec = replace(voice, language=lang) if lang else voice
            t1 = time.time(); audio, sr = engine.synthesize(text, spec); synth += time.time() - t1
            builder = builder or AudioBuilder(sample_rate=sr, crossfade_ms=50)
            builder.add_speech(audio, sr)
    path = out / f"{name.replace('+', '_')}.wav"
    sf.write(path, builder.build(), builder.sample_rate, subtype="PCM_16")
    vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    results.append({"engine": name, "load_s": round(load_s, 1), "audio_s": round(builder.duration_s, 1),
                    "synth_s": round(synth, 1), "rtf": round(synth / builder.duration_s, 2),
                    "peak_vram_gb": vram, "path": str(path)})
    print(json.dumps(results[-1]), flush=True)
    if name == "chatterbox_ml":
        cb = engine
    elif name == "chatterbox_ml+klon":
        cb.unload()
    else:
        engine.unload()
    torch.cuda.empty_cache()

stt = WhisperSTT("mobiuslabsgmbh/faster-whisper-large-v3-turbo"); stt.load("cuda")
for r in results:
    text, _ = stt.transcribe(Path(r["path"]), "id")
    r["wer"] = round(wer(ref_words, canon(text)), 3)
    r["transcript_head"] = text[:160]
summary = {"words": words, "ref_words_normalized": len(ref_words), "results": results}
Path("data/benchmark/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("SUMMARY", json.dumps(summary, ensure_ascii=False))
