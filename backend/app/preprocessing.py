"""Text cleaning and simple information extraction shared across the pipeline."""

from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Tuple

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


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_RANGE = re.compile(
    rf"{_MONTH}\s*'?(\d{{4}}|\d{{2}})\s*(?:-|–|—|to|till|until)\s*"
    rf"(?:{_MONTH}\s*'?(\d{{4}}|\d{{2}})|(present|current|now|date|ongoing))",
    re.IGNORECASE,
)


def _to_month_index(month: str, year: str) -> int:
    y = int(year)
    if y < 100:
        y += 2000
    return y * 12 + _MONTHS[month[:3].lower()] - 1


def _date_ranges(text: str, today: date) -> List[Tuple[int, int]]:
    """Find "Month YYYY - Month YYYY/Present" ranges as (start, end) month indexes."""
    now = today.year * 12 + today.month - 1
    ranges = []
    for m1, y1, m2, y2, present in _DATE_RANGE.findall(text):
        start = _to_month_index(m1, y1)
        end = now if present else _to_month_index(m2, y2)
        end = min(end, now)
        if start <= end:
            ranges.append((start, end + 1))  # inclusive of the last month
    return ranges


def years_from_date_ranges(text: str, today: Optional[date] = None) -> Optional[float]:
    """Total years covered by dated ranges, with overlapping periods merged.

    Only ranges that include a month name are used ("Jun 2025 - Aug 2025",
    "January 2026 - Present"). Bare year ranges such as "2023 - 2027" are
    ignored because they usually describe education.
    """
    ranges = sorted(_date_ranges(text, today or date.today()))
    if not ranges:
        return None
    total, (cur_start, cur_end) = 0, ranges[0]
    for start, end in ranges[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return round(total / 12, 1) if total else None


def extract_years_of_experience(text: str, today: Optional[date] = None) -> Optional[float]:
    """Estimate a candidate's (or a job's required) years of experience.

    1. The largest explicit "N years ... experience" phrase.
    2. Otherwise any "N years" mention.
    3. Otherwise the total length of dated ranges ("Jun 2025 - Present").

    Values above 40 are treated as noise and ignored.
    """
    if not text:
        return None
    values = [float(a or b) for a, b in _EXPERIENCE.findall(text)]
    if not values:
        values = [float(v) for v in _YEARS_ANYWHERE.findall(text)]
    values = [v for v in values if 0 < v <= 40]
    if values:
        return max(values)
    return years_from_date_ranges(text, today)
