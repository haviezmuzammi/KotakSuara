# Phase 4 — Laporan verifikasi (2026-10-11)

## 1. Perbandingan engine
Naskah uji sintetis yang sama untuk semua engine: 220 kata (294 kata setelah normalisasi angka/gelar),
berisi tanggal, tahun Hijriah, Rupiah, persen, jam, ordinal, gelar (SWT/SAW/r.a./rahimahumallah), istilah
Arab terkurasi, dan satu blok `[AYAT]`. Diukur di PC (RTX 4060 8 GB) dengan `scripts/benchmark.py`.

| engine | durasi audio | waktu sintesis | RTF | VRAM puncak | WER kasar* | komersial |
|---|---|---|---|---|---|---|
| `chatterbox_ml` (suara bawaan) | 123,6 dtk | 172,4 dtk | 1,40 | 3,59 GB | 0,279 | ✅ |
| `chatterbox_ml` + klon "Fahmi" | 132,8 dtk | 177,7 dtk | 1,34 | 3,83 GB | 0,276 | ✅ (dengan persetujuan) |
| `mms_id` | 127,9 dtk | 1,9 dtk | 0,01 | 0,51 GB | 0,286 | ❌ CC-BY-NC |
| `edge_id` | 103,7 dtk | 37,4 dtk | 0,36 | (cloud) | 0,262 | ❌ tidak resmi |
| `azure_id` | — | — | — | — | — | ✅ — **belum diuji: belum ada key** |

\* WER kasar = Whisper large-v3-turbo (bahasa `id`) dibandingkan dengan teks sumber setelah kedua sisi
dinormalisasi. Angkanya mirip untuk semua engine (0,26–0,29) karena sebagian besar "kesalahan" berasal dari
ejaan transliterasi Whisper (mis. "Salallahu" vs "shallallahu") dan istilah yang diucapkan dalam bahasa Arab —
**WER ini tidak membedakan kualitas engine**; penilaian utama tetap uji dengar tim.

Memori layanan (Chatterbox + Whisper dalam satu proses, diukur saat job + SRT): RAM ±2,1 GB saat aktif,
puncak 5,0 GB (saat model dimuat); GPU terpakai puncak ±3,7 GB dari 8 GB.

Perkiraan waktu proses Chatterbox (RTF 1,3–1,5): naskah 5 menit ≈ 7 menit, 15 menit ≈ 20–23 menit.

Penilaian dengar (pengguna): istilah Arab via model `ar` lebih baik daripada ejaan Latin via `ms`; kualitas
pelafalan keseluruhan ±80% — sisa perbaikan lewat fine-tune (ROADMAP B).

## 2. Klon suara
Chatterbox zero-shot dari voice note Telegram dengan persetujuan lisan (diverifikasi Whisper) — berjalan
(profil "Fahmi"). Profil tanpa kalimat persetujuan ditolak. OpenVoice tidak dipakai (lihat README).

## 3. Integrasi n8n (end-to-end, termasuk Wake-on-LAN)
Workflow `integrations/n8n/kotaksuara-narasi.json` (diimpor ke `n8n-fahmi` sebagai
"KotakSuara - Narasi (WoL, poll, unduh MP3)", credential Header Auth "KotakSuara Gateway").
Uji 2026-10-11 15:06: PC ditidurkan → workflow dijalankan → gateway mengirim WoL dan menjawab 503 →
workflow menunggu 60 dtk → 202 → 4× cek status → MP3 154 kB terunduh. Total 114 dtk.
Container `n8n-fahmi` juga dapat menjangkau layanan LAN langsung (`http://192.168.2.15:8780`, 401 tanpa key).

## 4. Reboot tanpa login
`shutdown /r` pukul 15:16:39 → boot 15:16:54 → `/health` 200 dari server ±21 dtk kemudian.
`quser`: tidak ada pengguna login; Windows Firewall aktif di semua profil; AutoAdminLogon tidak aktif.

## Definition of Done
| kriteria | status |
|---|---|
| `/health` menjawab setelah reboot tanpa login (LAN) | ✅ |
| semua tes lolos | ✅ 103 tes lokal + CI GitHub |
| laporan perbandingan engine | ✅ (Azure menunggu key) |
| n8n di `n8n-fahmi` menerima audio end-to-end termasuk WoL | ✅ |
| PC tidak ditahan bangun saat idle | ✅ tes execution-state; PC bisa ditidurkan saat idle |
| firewall hanya mengizinkan server | ✅ aturan 8780 hanya 192.168.1.60 + uji negatif |
| README: menambah engine & flag lisensi | ✅ |

Tidak dalam cakupan Tailscale (tailnet server bukan milik Yassalam); akses jarak jauh lewat C1/C2.
