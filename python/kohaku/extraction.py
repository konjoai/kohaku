"""Heuristic free-text → ``(subject, attribute, value)`` triple extraction.

`AnalogicalMemory` reasons over structured records, but agents accumulate *prose*
("The capital of France is Paris"). This module bridges the two with a small set
of **high-precision** regex patterns over common memory-statement forms, so the
relational reasoning works on what the agent actually read.

The design choice is precision over recall, deliberately: it is far better to
extract nothing from a sentence than to invent a wrong fact and pollute the
reasoning store (the Konjo way — no silent fabrication). Every triple carries a
confidence, and sentences that match no pattern yield nothing.

Patterns handled (case-insensitive, one triple per sentence clause):

* ``"The <attr> of <subject> is <value>"``        → (subject, attr, value)
* ``"<value> is the <attr> of <subject>"``        → (subject, attr, value)  [reversed]
* ``"<subject>'s <attr> is/: <value>"``           → (subject, attr, value)
* ``"<subject> is a/an <value>"``                 → (subject, "type", value)
* ``"<subject> likes/prefers/uses/... <value>"``  → (subject, <verb>, value)

Pure-Python, deterministic, no model call. For messy or novel phrasings, plug in
your own extractor and call :meth:`AnalogicalMemory.add_record` directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_ARTICLES = ("the ", "a ", "an ")
_PREFERENCE_VERBS = "likes|prefers|enjoys|loves|hates|dislikes|uses|needs|wants|avoids"

# Sentence/clause splitter — keep it simple and deterministic.
_CLAUSE_SPLIT = re.compile(r"[.!?\n]+|,? and | but |;")

_SUBJ = r"(?P<subj>[A-Za-z][\w .'-]*?)"
_ATTR = r"(?P<attr>[A-Za-z][\w -]*?)"
_VAL = r"(?P<val>[A-Za-z0-9][\w .'-]*?)"
_END = r"\s*$"


@dataclass(frozen=True)
class Triple:
    """An extracted ``(subject, attribute, value)`` fact with a confidence."""

    subject: str
    attribute: str
    value: str
    confidence: float


# (compiled pattern, attribute-group-or-None, confidence). When the attribute
# group is None the attribute is a fixed literal carried in the 4th slot.
_PATTERNS = [
    (
        re.compile(
            rf"^(?:the )?{_ATTR} of {_SUBJ} (?:is|are|was|were) {_VAL}{_END}", re.I
        ),
        None,
        None,
        0.9,
    ),
    (
        re.compile(rf"^{_VAL} (?:is|are|was|were) the {_ATTR} of {_SUBJ}{_END}", re.I),
        None,
        None,
        0.9,
    ),
    (
        re.compile(
            rf"^{_SUBJ}'s {_ATTR}(?:\s+(?:is|are|was|were)\s+|\s*:\s*){_VAL}{_END}",
            re.I,
        ),
        None,
        None,
        0.9,
    ),
    (
        re.compile(rf"^{_SUBJ} (?:is|are) (?:a|an) {_VAL}{_END}", re.I),
        None,
        "type",
        0.8,
    ),
    (
        re.compile(rf"^{_SUBJ} (?P<verb>{_PREFERENCE_VERBS}) {_VAL}{_END}", re.I),
        "verb",
        None,
        0.75,
    ),
]


def _normalise(token: str) -> str:
    """Lowercase, strip a leading article and surrounding punctuation/space."""
    t = token.strip().strip("\"'.,;:").strip().lower()
    for art in _ARTICLES:
        if t.startswith(art):
            t = t[len(art) :]
            break
    return t.strip()


def _match_clause(clause: str) -> Optional[Triple]:
    """Return the triple for the first pattern that matches ``clause``, else None.

    A pattern matching but normalising to an empty part counts as the match (we
    stop scanning) yet yields nothing — the clause is claimed but unparseable.
    """
    for pattern, attr_group, fixed_attr, conf in _PATTERNS:
        m = pattern.match(clause)
        if not m:
            continue
        groups = m.groupdict()
        attr = fixed_attr or groups.get(attr_group or "attr")
        subject = _normalise(groups["subj"])
        attribute = _normalise(attr or "")
        value = _normalise(groups["val"])
        if subject and attribute and value:
            return Triple(subject, attribute, value, conf)
        return None
    return None


def extract_triples(text: str) -> List[Triple]:
    """Extract high-precision ``(subject, attribute, value)`` triples from ``text``.

    Splits on sentence/clause boundaries and applies each pattern to each clause,
    returning the first match per clause. Returns ``[]`` when nothing parses.
    """
    triples: List[Triple] = []
    for clause in _CLAUSE_SPLIT.split(text):
        clause = clause.strip()
        if not clause:
            continue
        triple = _match_clause(clause)
        if triple is not None:
            triples.append(triple)
    return triples


def records_from_texts(texts: List[str]) -> Dict[str, Dict[str, str]]:
    """Group triples extracted from many texts into ``{subject: {attr: value}}``.

    Later mentions overwrite earlier ones for the same ``(subject, attribute)``.
    """
    records: Dict[str, Dict[str, str]] = {}
    for text in texts:
        for triple in extract_triples(text):
            records.setdefault(triple.subject, {})[triple.attribute] = triple.value
    return records
