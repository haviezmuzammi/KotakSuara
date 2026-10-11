# Roadmap: klon suara, model Indonesia, dan akses agen cloud

Status per 2026-10-11. Rincian teknis untuk agen ada di `CLAUDE.md`.

## A. Klon suara lewat bot Telegram — SELESAI
Bot `@yassalam_monitor_bot` (repo terpisah di server) memakai API KotakSuara:

| Perintah | Fungsi |
|---|---|
| `/klon_suara <nama>` lalu kirim voice note 15–60 dtk | Voice note **wajib dibuka kalimat persetujuan** ("Saya … mengizinkan suara saya digunakan untuk narasi Yassalam"). Bot mentranskrip dengan Whisper; tanpa kata izin + "suara", profil ditolak. File asli disimpan sebagai bukti persetujuan. |
| `/suara`, `/suara <nomor>` | Daftar & pilih suara aktif (0 = bawaan) |
| `/narasi <teks>` atau file `.txt` dengan caption `/narasi` | Narasi MP3 dengan suara aktif + daftar posisi jeda ayat |
| `/hapus_suara <nomor>` | Hapus profil beserta rekamannya |

Bot menyalakan PC lewat Wake-on-LAN bila layanan belum aktif.
Catatan: klon zero-shot meniru **warna suara**; logat masih condong Melayu (model `ms`). Untuk logat Indonesia → B.

## B. Model Indonesia sendiri (fine-tune Chatterbox)

Pendekatan yang disarankan untuk **beberapa narator**: latih **satu adaptor LoRA "logat Indonesia"** dari
gabungan rekaman semua narator, lalu tiap narator tetap memakai klon zero-shot di atasnya. Hasilnya:
pelafalan Indonesia membaik untuk semua suara, tanpa melatih model per orang. Bila satu narator utama
ingin kualitas maksimal, tambahkan LoRA khusus dia (≥ 1 jam data).

| Tahap | Isi | Target |
|---|---|---|
| B1 Data | Naskah rekaman (kalimat beragam fonem Indonesia + istilah kajian), dibaca tiap narator; WAV ≥ 22 kHz mono, ruang senyap; **persetujuan tertulis/lisan per narator** | 30 menit minimal per narator, total 2–5 jam |
| B2 Persiapan | Potong per kalimat 3–10 dtk (VAD), transkrip Whisper + `normalizer_id`, cek manual; format LJSpeech `file|teks|teks_normal` | data tidak pernah masuk repo publik |
| B3 Latih | LoRA Chatterbox Multilingual (toolkit kandidat di bawah), mulai dari bahasa `ms` | beberapa jam GPU |
| B4 Evaluasi | WER (Whisper) vs teks, uji dengar tim (A/B dengan model dasar), cek ayat tetap jeda | WER turun, tim lebih suka |
| B5 Pakai | Engine baru `chatterbox_id` di `app/engines/`, `commercial_ok=True` (bobot dasar MIT + data milik Yassalam ber-persetujuan) | |

Toolkit kandidat (verifikasi lisensi & kecocokan versi sebelum dipakai):
- [gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning) — Apache-2.0, LoRA & full fine-tune, format LJSpeech, ekstensi vocabulary, minimal ±30 menit data.
- [rkumar1310/indic-chatter-train](https://github.com/rkumar1310/indic-chatter-train) — contoh menambah bahasa baru via ekstensi tokenizer + LoRA (bahasa India).
- [Ahmed-Ezzat20/chatterbox-finetuning-multilingual](https://github.com/Ahmed-Ezzat20/chatterbox-finetuning-multilingual) — skrip `lora.py` untuk model multilingual.

### Sewa GPU (RTX 4060 8 GB kemungkinan kurang untuk melatih)
Harga dicek 2026-10-11 dari halaman resmi, bisa berubah:

| Penyedia | GPU | Harga/jam | Catatan |
|---|---|---|---|
| [RunPod](https://www.runpod.io/pricing) Secure Cloud | RTX 4090 24 GB | $0.74 | **disarankan**: pusat data tersertifikasi, cocok untuk data suara pribadi; tagihan per detik |
| RunPod Secure Cloud | A40 48 GB | $0.49 | VRAM besar, lebih murah dari 4090 tapi lebih lambat |
| RunPod Community Cloud | RTX 4090 / A40 | $0.34 / $0.35 | lebih murah, tapi host milik perorangan — **hindari untuk rekaman narator** |
| [Vast.ai](https://vast.ai/pricing) | beragam | pasar dinamis | tagihan per detik, opsi interruptible >50% lebih murah; host pihak ketiga |

Perkiraan: LoRA dari 2–5 jam data ≈ 2–6 jam GPU → **±$2–5 di RTX 4090 Secure Cloud**. Hapus volume/pod setelah
selesai; unduh hanya bobot LoRA hasil latihan.

## C. Akses untuk agen Claude di cloud

### C1. Antrean job lewat GitHub (direncanakan, utama)
- Repo **privat** `KotakSuara-jobs`. Agen menulis `jobs/<id>.json` (`text`, `engine`, `profile`), commit/PR.
- Poller di server (layanan systemd seperti bot) cek tiap ±30 dtk → WoL bila perlu → `/v1/tts` → unggah
  MP3 + SRT sebagai rilis/berkas di repo privat → tulis `results/<id>.json`.
- Tidak ada port terbuka ke internet; hak akses = izin GitHub; jejak audit otomatis. Latensi ±1–2 menit + waktu proses.

### C2. Tunnel terautentikasi (opsi lanjutan)
- `cloudflared` di **server** (bukan PC) → gateway kecil yang meneruskan ke KotakSuara di LAN.
- Wajib: Cloudflare Access service token (atau mTLS) + `X-API-Key`; rate limit; hanya endpoint TTS/job.
- Membutuhkan domain di Cloudflare dan izin jaringan sandbox agen cloud ke domain itu.
- Dapat dibungkus sebagai **MCP server** agar agen claude.ai memakai TTS sebagai tool.
- Lebih cepat (real-time) tetapi menambah permukaan serangan; aktifkan hanya bila C1 tidak cukup.
