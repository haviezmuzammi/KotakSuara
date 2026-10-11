"""Indonesian number-to-words conversion."""
from __future__ import annotations

_UNITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
_SCALES = [
    (10**12, "triliun"),
    (10**9, "miliar"),
    (10**6, "juta"),
    (10**3, "ribu"),
]


def _below_thousand(n: int) -> str:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        words.append("seratus")
    elif hundreds > 1:
        words.append(f"{_UNITS[hundreds]} ratus")
    if rest == 0:
        pass
    elif rest < 10:
        words.append(_UNITS[rest])
    elif rest == 10:
        words.append("sepuluh")
    elif rest == 11:
        words.append("sebelas")
    elif rest < 20:
        words.append(f"{_UNITS[rest - 10]} belas")
    else:
        tens, ones = divmod(rest, 10)
        words.append(f"{_UNITS[tens]} puluh")
        if ones:
            words.append(_UNITS[ones])
    return " ".join(words)


def number_to_words(n: int) -> str:
    """Convert a non-negative integer to Indonesian words (up to < 10**15)."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n == 0:
        return "nol"
    if n >= 10**15:
        return digits_to_words(str(n))
    words: list[str] = []
    for value, name in _SCALES:
        count, n = divmod(n, value)
        if count:
            # "seribu" but "satu juta", "satu miliar"
            words.append("seribu" if (value == 1000 and count == 1) else f"{number_to_words(count)} {name}")
    if n:
        words.append(_below_thousand(n))
    return " ".join(words)


def digits_to_words(digits: str) -> str:
    """Read a digit string digit by digit (phone numbers, very long numbers)."""
    return " ".join(_UNITS[int(d)] for d in digits if d.isdigit())


def decimal_to_words(integer_part: str, fraction_part: str | None = None) -> str:
    """'2', '5' -> 'dua koma lima'. Fraction digits are read one by one, as is usual in Indonesian."""
    words = number_to_words(int(integer_part))
    if fraction_part:
        words += " koma " + digits_to_words(fraction_part)
    return words


def ordinal_to_words(n: int) -> str:
    """1 -> 'pertama', 2 -> 'kedua', 21 -> 'kedua puluh satu'."""
    if n == 1:
        return "pertama"
    return "ke" + number_to_words(n)
