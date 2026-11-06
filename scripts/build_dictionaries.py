"""Step 1b: pronunciation dictionaries for the forced aligner (RECONSTRUCTION of the notebook cell that was run on Kaggle; same logic).

English : MFA's `english_us_arpa` dictionary + MFA G2P pronunciations for corpus words that are missing from it (1,173 of 6,687 distinct words,
          4.23% of tokens)  ->  english_combined.dict
Telugu  : graphemic dictionary. Every distinct transcript word (already ISO 15919, see telugu_text.py) is mapped to its symbol sequence
          (letters; aspirates as single symbols)  ->  telugu.dict

Usage (from the working directory that holds raw_data/):  python scripts/build_dictionaries.py
Requires the MFA conda environment on PATH for the English part (mfa g2p)."""
import glob, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Telugu'))

def words(lang):
    ws = set()
    for f in glob.glob(f"raw_data/{lang}/*/*.lab"):
        ws.update(open(f, encoding="utf-8").read().split())
    return sorted(ws)

def english():
    mfa_dict = os.path.expanduser("~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict")
    known = {l.split()[0] for l in open(mfa_dict, encoding="utf-8") if l.strip()}
    ws = words("English")
    oov = [w for w in ws if w not in known]
    print(f"{len(ws)} distinct words, {len(oov)} out of vocabulary")
    open("english_oov.txt", "w").write("\n".join(oov) + "\n")
    subprocess.run("mfa g2p english_oov.txt english_us_arpa english_oov.dict --clean", shell=True, check=True)
    with open("english_combined.dict", "w", encoding="utf-8") as out:
        out.write(open(mfa_dict, encoding="utf-8").read().rstrip("\n") + "\n")
        out.write(open("english_oov.dict", encoding="utf-8").read())

def telugu():
    import telugu_text as TT
    with open("telugu.dict", "w", encoding="utf-8") as out:
        ws = words("Telugu")
        for w in ws:
            out.write(w + "\t" + " ".join(TT.iso_to_symbols(w, strict=True)) + "\n")
    print(len(ws), "Telugu dictionary words")

if __name__ == "__main__":
    english(); telugu()
