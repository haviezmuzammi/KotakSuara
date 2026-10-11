# Naskah rekaman narator (B1)

`naskah_id.tsv` — 203 kalimat (±1.465 kata, ±13 menit bacaan) dalam 7 kategori:

| kategori | isi |
|---|---|
| `umum` | kalimat sehari-hari yang mencakup bunyi Indonesia (ng, ny, sy, kh, z, f, v, diftong, gugus konsonan) + kalimat tanya/seru |
| `angka` | jam, tanggal, Rupiah, persen, nama bulan Hijriah |
| `asmaul_husna` | nama-nama Allah dalam kalimat, dengan makna Indonesia yang lazim |
| `gelar` | nabi ('alaihissalam / 'alaihimassalam / 'alaihimussalam), sahabat (radhiyallahu 'anhu / 'anha / 'anhuma / 'anhum), ulama (rahimahullah / rahimahumallah / rahimahumullah) |
| `istilah` | serapan Arab yang umum di Indonesia: akidah, tauhid, sunnah, bid'ah, ittiba', ijmak, qiyas, mulazamah, iktikaf, qana'ah, istiqamah, … |
| `dzikir` | salam, basmalah, hamdalah, tasbih, istirja', kafaratul majelis, dll. dalam transliterasi |
| `kajian` | kalimat pembuka/penutup/peralihan khas kajian |

Tidak ada ayat Al-Qur'an di naskah ini (kebijakan proyek: ayat tidak pernah disintesis).

**Sebelum merekam:** mohon naskah ditinjau oleh asatidzah Yassalam (ejaan transliterasi, makna Asmaul Husna,
pemakaian gelar). Koreksi cukup diedit di file TSV ini.

## Target data
- Minimal **30 menit per narator**: baca naskah ini satu kali (±13 menit) + ±20 menit naskah kajian Yassalam
  sendiri (paling cocok dengan gaya yang ingin dihasilkan). Total semua narator 2–5 jam.
- Gaya: narasi kajian yang tenang dan jelas, tempo biasa. Jangan dibuat-buat; ulangi kalimat bila tersandung.

## Cara merekam
- Ruang senyap tanpa gema (kamar dengan kasur/lemari pakaian lebih baik daripada aula). Matikan kipas/AC.
- Mikrofon USB/lavalier yang layak, jarak ±15–20 cm, posisi tetap. Hindari HP di meja yang bergetar.
- Format: **WAV, mono, 44,1 atau 48 kHz, 16/24-bit**. Jangan MP3/WhatsApp (kompresi merusak data latih).
- Satu file per kalimat dengan nama `<id>.wav` (mis. `S012.wav`), ±0,5 dtk hening di awal & akhir.
  Bila lebih mudah merekam panjang, rekam per kategori; pemotongan otomatis dilakukan di tahap B2.
- Simpan di PC: `D:\AI\datasets\narator\<nama-narator>\` — **jangan pernah diunggah ke repo publik**.

## Persetujuan (wajib, satu per narator)
Rekam satu file `persetujuan.wav` di folder narator berisi pernyataan, misalnya:

> "Saya [nama lengkap] mengizinkan Yayasan Amal Sahabat Islam (Yassalam) merekam dan menggunakan suara saya
> untuk melatih dan menjalankan model suara narasi kajian Yassalam. Saya dapat mencabut izin ini kapan saja."

Catat juga di `persetujuan.txt`: nama, tanggal, kontak, dan cakupan izin. Bila izin dicabut, data dan bobot
model yang dilatih dari suara tersebut dihapus.

## Catatan teknis
Teks latih harus melewati `normalizer_id` yang sama dengan saat inferensi (angka → kata, SWT → "subhanahu wa
ta'ala", dll.) agar model belajar pasangan teks–suara yang konsisten. Entri `pronunciations` di
`config/lexicon.yaml` (mis. shalat → solat untuk model `ms`) perlu dievaluasi ulang setelah fine-tune.
