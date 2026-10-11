import os
import time

import pytest

from app.config import REPO_ROOT
from app.text.chunker import chunk_text
from app.text.lexicon import Lexicon, LexiconData, parse_lexicon
from app.text.normalizer_id import apply_symbols, normalize
from app.text.numbers_id import decimal_to_words, digits_to_words, number_to_words, ordinal_to_words
from app.text.segmenter import AyatTagError, split_segments


@pytest.fixture(scope="module")
def lex() -> LexiconData:
    return Lexicon(REPO_ROOT / "config" / "lexicon.yaml").get()


# numbers ------------------------------------------------------------------
@pytest.mark.parametrize("n, words", [
    (0, "nol"), (7, "tujuh"), (10, "sepuluh"), (11, "sebelas"), (15, "lima belas"),
    (20, "dua puluh"), (99, "sembilan puluh sembilan"), (100, "seratus"), (101, "seratus satu"),
    (250, "dua ratus lima puluh"), (1000, "seribu"), (1447, "seribu empat ratus empat puluh tujuh"),
    (2026, "dua ribu dua puluh enam"), (11000, "sebelas ribu"), (100000, "seratus ribu"),
    (1_000_000, "satu juta"), (2_500_000, "dua juta lima ratus ribu"),
    (3_000_000_000, "tiga miliar"), (1_000_000_000_000, "satu triliun"), (-5, "minus lima"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_number_helpers():
    assert number_to_words(10**15) == digits_to_words("1" + "0" * 15)
    assert decimal_to_words("2", "5") == "dua koma lima"
    assert decimal_to_words("3") == "tiga"
    assert ordinal_to_words(1) == "pertama"
    assert ordinal_to_words(3) == "ketiga"


# normalizer ---------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("Infak Rp 50.000,- saja.", "Infak lima puluh ribu rupiah saja."),
    ("Biaya Rp5 juta.", "Biaya lima juta rupiah."),
    ("Naik 2,5% tahun ini.", "Naik dua koma lima persen tahun ini."),
    ("Tanggal 10/10/2026 kita kajian.", "Tanggal sepuluh Oktober dua ribu dua puluh enam kita kajian."),
    ("Tahun 1447 H.", "Tahun seribu empat ratus empat puluh tujuh Hijriah."),
    ("Tahun 2026 M bertepatan.", "Tahun dua ribu dua puluh enam Masehi bertepatan."),
    ("Wafat 11 H di Madinah.", "Wafat sebelas Hijriah di Madinah."),
    ("Dibangun 500 SM.", "Dibangun lima ratus sebelum Masehi."),
    ("Pukul 04.30 WIB.", "Pukul empat tiga puluh waktu Indonesia bagian barat."),
    ("jam 7.00 pagi.", "jam tujuh pagi."),
    ("Bab ke-3 dan ke2.", "Bab ketiga dan kedua."),
    ("Halaman 10-12.", "Halaman sepuluh sampai dua belas."),
    ("Ada 1.250 santri.", "Ada seribu dua ratus lima puluh santri."),
    ("Nilainya 3.75 poin.", "Nilainya tiga koma tujuh lima poin."),
    ("Telepon 0812345.", "Telepon nol delapan satu dua tiga empat lima."),
    ("Lihat QS. Al-Baqarah: 183.", "Lihat Al-Qur'an surah Al-Baqarah ayat seratus delapan puluh tiga."),
    ("Lihat QS Ali Imran: 102-103.", "Lihat Al-Qur'an surah Ali Imran ayat seratus dua sampai seratus tiga."),
    ("Lihat QS 2:255.", "Lihat Al-Qur'an surah nomor dua ayat dua ratus lima puluh lima."),
    ("Lihat QS 2:1-5.", "Lihat Al-Qur'an surah nomor dua ayat satu sampai lima."),
    ("Nabi SAW bersabda (HR. Muslim).", "Nabi shallallahu 'alaihi wa sallam bersabda, hadits riwayat Muslim."),
    ("Allah SWT berfirman.", "Allah subhanahu wa ta'ala berfirman."),
    ("Shalat itu tiang agama.", "Solat itu tiang agama."),
    ("Insya Allah besok.", "Insya-Allah besok."),
    ("Kitab, buku, dll.", "Kitab, buku, dan lain-lain."),
    ("Rasulullah ﷺ bersabda.", "Rasulullah shallallahu 'alaihi wa sallam bersabda."),
    ("Ustadz & santri", "Ustadz dan santri."),
    ("## Judul\nIsi paragraf", "Judul. Isi paragraf."),
    ("Pada 5 M bukan tahun.", "Pada lima M bukan tahun."),
    ("Tanggal 45/13/2026.", "Tanggal empat puluh lima/tiga belas/dua ribu dua puluh enam."),
])
def test_normalize(text, expected, lex):
    assert normalize(text, lex) == expected


def test_apply_symbols(lex):
    assert "shallallahu" in apply_symbols("Nabi ﷺ", lex)


# segmenter ----------------------------------------------------------------
def test_segments_ayat_tag_and_arabic():
    segs = split_segments("Allah berfirman: [AYAT]يَا أَيُّهَا الَّذِينَ آمَنُوا[/AYAT] Artinya: wahai orang beriman. "
                          "Doa: رَبَّنَا آتِنَا فِي الدُّنْيَا حَسَنَةً selesai.")
    assert [s.kind for s in segs] == ["speech", "ayat", "speech", "ayat", "speech"]
    assert segs[1].source == "tag" and segs[3].source == "arabic"
    assert segs[3].text.startswith("رَبَّنَا")


def test_segments_tag_case_and_punctuation_only_dropped():
    segs = split_segments("[ayat]text[/Ayat].")
    assert [(s.kind, s.text) for s in segs] == [("ayat", "text")]


@pytest.mark.parametrize("bad", ["[AYAT] tidak ditutup", "[/AYAT] liar", "[AYAT][AYAT]x[/AYAT][/AYAT]"])
def test_segments_bad_tags(bad):
    with pytest.raises(AyatTagError):
        split_segments(bad)


def test_symbol_not_silenced_after_apply(lex):
    segs = split_segments(apply_symbols("Rasulullah ﷺ bersabda", lex))
    assert [s.kind for s in segs] == ["speech"]


# chunker ------------------------------------------------------------------
def test_chunker_packs_sentences():
    text = "Kalimat satu. Kalimat dua! Kalimat tiga? " * 3
    chunks = chunk_text(text, max_chars=60)
    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_chunker_splits_long_sentence():
    long = "kata " * 80 + "akhir, lalu klausa pendek."
    chunks = chunk_text(long, max_chars=50)
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks).split() == long.split()


def test_chunker_rejects_tiny_limit():
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=5)


# lexicon ------------------------------------------------------------------
def test_lexicon_hot_reload_and_invalid_file(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text("abbreviations:\n  ABC: satu\n", encoding="utf-8")
    lexicon = Lexicon(path)
    assert lexicon.get().abbreviations == {"ABC": "satu"}

    path.write_text("abbreviations:\n  ABC: dua\n", encoding="utf-8")
    later = time.time() + 5
    os.utime(path, (later, later))
    assert lexicon.get().abbreviations == {"ABC": "dua"}

    path.write_text("abbreviations: [broken", encoding="utf-8")
    os.utime(path, (later + 5, later + 5))
    assert lexicon.get().abbreviations == {"ABC": "dua"}
    assert lexicon.last_error


def test_lexicon_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Lexicon(tmp_path / "nope.yaml")


@pytest.mark.parametrize("bad", ["- a\n- b\n", "symbols: [1, 2]\n", "abbreviations:\n  X: 5\n"])
def test_parse_lexicon_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        parse_lexicon(bad)


def test_lexicon_deleted_after_load_keeps_data(tmp_path):
    path = tmp_path / "lex.yaml"
    path.write_text("symbols:\n  '@': di\n", encoding="utf-8")
    lexicon = Lexicon(path)
    path.unlink()
    assert lexicon.get().symbols == {"@": "di"}
    assert lexicon.last_error


def test_new_honorific_abbreviations(lex):
    assert normalize("Nabi Ibrahim a.s. dan Abu Bakar r.a.", lex) == \
        "Nabi Ibrahim 'alaihissalam dan Abu Bakar radhiyallahu 'anhu."
    assert normalize("Layanan KotakSuara siap.", lex) == "Layanan Kotak Suara siap."


def test_recording_script_is_clean():
    import csv
    import re

    rows = list(csv.DictReader((REPO_ROOT / "training" / "naskah" / "naskah_id.tsv").open(encoding="utf-8"),
                               delimiter="\t"))
    assert len(rows) >= 200
    assert len({r["id"] for r in rows}) == len(rows)
    assert not any(re.search(r"[\u0600-\u06FF]", r["teks"]) for r in rows)  # no Arabic script / verses
