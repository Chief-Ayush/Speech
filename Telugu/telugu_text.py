"""Telugu text frontend for the FastSpeech 2 Telugu model.

Pipeline:  raw Telugu text -> normalise -> expand digits to Telugu words -> ISO 15919 romanisation
           (Aksharamukha) -> list of model symbols (letters, aspirate digraphs, 'sp' for punctuation).

Design notes (see Report):
  * ISO 15919 is a published standard, so no pronunciation judgement calls are made by us. It spells out the
    inherent 'a' of each consonant, marks long vowels (ā ī ū ē ō), retroflex consonants (ṭ ḍ ṇ ḷ) and the
    anusvara (ṁ), and writes conjunct/geminate consonants by repeating the letter (paṭṭu).
  * Word boundaries are NOT encoded as symbols (same as the reference English pipeline); punctuation becomes 'sp'.
  * Aspirated consonants are two ISO letters (k+h); we treat them as ONE symbol (kh gh ch jh ṭh ḍh th dh ph bh).
  * Digits are expanded with num2words (lang='te') plus small documented fixes; every number rule is a
    candidate for review because it could not be checked by a native speaker (verified with ASR instead).
"""
import re
import unicodedata

SP = "sp"
PAUSE_CHARS = set(",.?!;:")
DIGRAPHS = {"kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh"}

# All symbols the model can receive. Anything else raises an error (we never silently drop characters).
BASE_SYMBOLS = (
    list("aāiīuūṛṝeēoō")                      # vowels (ai, au are written as a+i, a+u by ISO)
    + list("kgcjṭḍtdpbṅñṇnmyrlḷvśṣshṟ")        # consonants
    + sorted(DIGRAPHS)                          # aspirates
    + list("ṁḥ")                                # anusvara, visarga
    + [SP]
)

_ZERO_WIDTH = "​‌‍﻿⁠"
_QUOTES = "\"“”‘’'`´"


def _num_words(n: int) -> str:
    from num2words import num2words
    s = num2words(n, lang="te")
    # num2words writes the thousands multiplier as 'వేయిల' (e.g. 2026); natural written Telugu uses 'వేల'.
    # (Candidate fix, to be confirmed by the ASR round-trip check.)
    s = s.replace("వేయిల", "వేల")
    return s


def _ordinal_suffix(words: str) -> str:
    """12వ -> 'పన్నెండవ': a final 'ు' is replaced by 'వ'; otherwise 'వ' is appended."""
    return words[:-1] + "వ" if words.endswith("ు") else words + "వ"


def expand_numbers(text: str) -> str:
    # times first: 10:45 -> '<hours> <minutes>'  (assumption: spoken as two numbers; check with ASR)
    text = re.sub(r"(\d{1,2}):(\d{2})",
                  lambda m: _num_words(int(m.group(1))) + " " + _num_words(int(m.group(2))), text)
    # ordinals: 12వ
    text = re.sub(r"(\d+)\s*వ(?![ఀ-౿])",
                  lambda m: _ordinal_suffix(_num_words(int(m.group(1)))), text)
    # plain integers
    text = re.sub(r"\d+", lambda m: _num_words(int(m.group(0))), text)
    return text


def normalize_script(text: str) -> str:
    """Clean Telugu script text (no number expansion)."""
    t = unicodedata.normalize("NFC", text)
    for ch in _ZERO_WIDTH:
        t = t.replace(ch, "")
    for ch in _QUOTES:
        t = t.replace(ch, "")
    t = t.replace(";", ",").replace(":", ",")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def to_iso(text: str) -> str:
    from aksharamukha import transliterate
    iso = transliterate.process("Telugu", "ISO", text)
    iso = unicodedata.normalize("NFC", iso)
    # vocalic r is r + combining ring below; encode as one symbol so it is not split into two tokens
    iso = iso.replace("r̥̄", "ṝ").replace("r̥", "ṛ")
    return iso


def iso_to_symbols(iso: str, strict: bool = True):
    out, i = [], 0
    allowed = set(BASE_SYMBOLS)
    while i < len(iso):
        ch = iso[i]
        if ch.isspace():
            i += 1
            continue
        if ch in PAUSE_CHARS:
            if not out or out[-1] != SP:
                out.append(SP)
            i += 1
            continue
        if i + 1 < len(iso) and iso[i:i + 2] in DIGRAPHS:
            out.append(iso[i:i + 2])
            i += 2
            continue
        if strict and ch not in allowed:
            raise ValueError("Unmapped symbol %r (U+%04X) in %r" % (ch, ord(ch), iso))
        out.append(ch)
        i += 1
    while out and out[-1] == SP:
        out.pop()
    while out and out[0] == SP:
        out.pop(0)
    return out


def text_to_symbols(text: str, strict: bool = True):
    t = expand_numbers(text)
    t = normalize_script(t)
    return iso_to_symbols(to_iso(t), strict=strict)


def text_to_lab(text: str) -> str:
    """Word-level ISO text without punctuation (corpus transcript for the forced aligner)."""
    t = normalize_script(expand_numbers(text))
    iso = to_iso(t)
    iso = "".join(" " if c in PAUSE_CHARS else c for c in iso)
    return re.sub(r"\s+", " ", iso).strip()


def self_test():
    tests = [
        "పులి అడ  విలో నిశ్శబ్దంగా నడిచింది.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
    ]
    for t in tests:
        print(t)
        print("  expanded:", normalize_script(expand_numbers(t)))
        print("  symbols :", " ".join(text_to_symbols(t)))
    for n in (0, 1, 2, 10, 11, 12, 20, 45, 100, 1000, 2000, 2026):
        print(n, "->", _num_words(n))


if __name__ == "__main__":
    self_test()
