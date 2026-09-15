import re
from html import unescape

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return unescape(_TAG_RE.sub(" ", text or "")).strip()
