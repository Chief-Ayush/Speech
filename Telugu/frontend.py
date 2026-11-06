"""Telugu text front-end: normalisation -> number expansion -> ISO 15919 romanisation -> symbol sequence (see telugu_text.py)."""
from telugu_text import BASE_SYMBOLS as SYMBOLS, text_to_symbols  # noqa: F401


def text_to_phones(text, verbose=False):
    return text_to_symbols(text)
