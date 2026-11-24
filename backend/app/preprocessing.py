"""Text cleaning and simple information extraction shared across the pipeline."""

from __future__ import annotations

import re
from typing import Optional

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
# Keep characters that matter in tech terms: c++, c#, node.js, ci/cd
_NON_TEXT = re.compile(r"[^a-z0-9+#./\s-]")
_LOOSE_PUNCT = re.compile(r"(?<![a-z0-9])[./-]|[./-](?![a-z0-9])")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case the text and strip URLs, emails, phone numbers and noise.

    Characters such as ``+``, ``#`` and ``.`` are kept inside words so that
    terms like ``C++``, ``C#`` and ``Node.js`` survive cleaning.
    """
    if not text:
        return ""
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    text = _NON_TEXT.sub(" ", text)
    text = _LOOSE_PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


_EXPERIENCE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b(?:\s+\w+){0,4}?\s+experience"
    r"|experience\s+(?:of\s+)?(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_YEARS_ANYWHERE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)


def extract_years_of_experience(text: str) -> Optional[float]:
    """Return the largest "N years ... experience" value found in the text.

    Falls back to any "N years" mention when no explicit experience phrase
    exists. Values above 40 are treated as noise and ignored.
    """
    if not text:
        return None
    values = [float(a or b) for a, b in _EXPERIENCE.findall(text)]
    if not values:
        values = [float(v) for v in _YEARS_ANYWHERE.findall(text)]
    values = [v for v in values if 0 < v <= 40]
    return max(values) if values else None
