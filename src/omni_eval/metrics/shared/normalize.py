"""Text normalization rules fixed for metric version 1."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    return " ".join(value.split())
