"""English text front-end: text normalisation -> ARPAbet phones (CMUdict + g2p_en for out-of-vocabulary words).
Training targets come from MFA (english_us_arpa); the symbol inventory is the same ARPAbet set with stress digits."""
import re

from num2words import num2words

VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
SP = "sp"
SYMBOLS = [v + s for v in VOWELS for s in "012"] + CONSONANTS + [SP]

LETTERS = {"@a": ["EY1"], "@m": ["EH1", "M"], "@p": ["P", "IY1"]}   # spelled-out letters for AM / PM
MONTHS = {m: m for m in "january february march april may june july august september october november december".split()}
PUNCT = ",.;:?!"

_g2p = None


def g2p():
    global _g2p
    if _g2p is None:
        import nltk
        for r, p in (("averaged_perceptron_tagger_eng", "taggers"), ("cmudict", "corpora")):
            try:
                nltk.data.find(f"{p}/{r}")
            except LookupError:
                nltk.download(r, quiet=True)
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


def normalize(text):
    """Spoken-form expansion of the numerals in test sentence E4 (times, years, day numbers); lower-case."""
    t = text.replace("’", "'")
    t = re.sub(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]\.?",
               lambda m: _time(int(m.group(1)), int(m.group(2))) + " " + "@" + m.group(3).lower() + " @m", t)
    t = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", lambda m: num2words(int(m.group(1)), to="year"), t)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group(0))), t)
    return t.replace("-", " ").lower()


def _time(h, m):
    if m == 0:
        return num2words(h)
    return num2words(h) + " " + ("oh " + num2words(m) if m < 10 else num2words(m))


def text_to_phones(text, verbose=False):
    """Words are converted PHRASE by phrase (a run of words between punctuation marks / spelled-out letters), not one word at a time,
    so that g2p_en's part-of-speech tagger can choose context-dependent pronunciations (noun CON-duct vs verb con-DUCT)."""
    t = normalize(text)
    if verbose:
        print("normalised:", t)
    toks = re.findall(r"@[apm]|[a-z']+|[,.;:?!]", t)
    out, run = [], []

    def flush():
        if run:
            ph = [p for p in g2p()(" ".join(run)) if p != " "]
            out.extend(p for p in ph if p in SYMBOLS)
            run.clear()

    for tok in toks:
        if tok in PUNCT:
            flush()
            if out and out[-1] != SP:
                out.append(SP)
        elif tok in LETTERS:
            flush()
            out.extend(LETTERS[tok])
        else:
            run.append(tok)
    flush()
    while out and out[-1] == SP:
        out.pop()
    return out


if __name__ == "__main__":
    for s in ["The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."]:
        print(" ".join(text_to_phones(s, True)))
