"""Declared Unicode text normalization used only by normalized metrics."""

import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall("\\w+", text, flags=re.UNICODE))
