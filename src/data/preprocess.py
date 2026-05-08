"""
preprocess.py  –  Amharic text normalisation for AfriAware.
"""

import re
import unicodedata

URL_RE       = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE   = re.compile(r"@\w+")
HASHTAG_RE   = re.compile(r"#(\w+)")
EMOJI_RE     = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_amharic(text: str) -> str:
    """Light normalisation that preserves Ethiopic script."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = EMOJI_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()
