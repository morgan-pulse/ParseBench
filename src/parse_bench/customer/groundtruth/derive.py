"""Derive evaluation rules from a reference transcription.

Every rule here is produced by deterministic code, not by a model. The bags are
built with the evaluator's own tokenizers (``rules_bag``) so that expected and
actual text are normalized identically — a rule can never fail because the two
sides disagreed about what a word is.

Rule types are chosen to feed the metrics the reports key off:

* ``content_faithfulness`` = ``normalized_text_correctness`` (the ``*_percent``
  bag rules) + ``normalized_order`` (``order`` rules), per
  ``evaluators/parse.py``.
* ``semantic_formatting`` = ``normalized_text_styling`` (bold / strikeout /
  sup / sub, positive and negative) + ``normalized_title_accuracy``
  (``is_title``, ``title_hierarchy_percent``).
* Table metrics (GriTS / TEDS / TableRecordMatch) read the reference markdown
  directly, so the ``table`` category needs no rules at all.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from parse_bench.evaluation.metrics.parse.rules_bag import (
    SentenceBagRule,
    WordBagRule,
    _extract_digit_counts,
)

# Reused so ordered sentence extraction matches the evaluator's pipeline exactly.
from parse_bench.evaluation.metrics.parse.rules_base import (  # noqa: PLC2701
    _strip_and_replace_latex,
    _strip_fenced_code_blocks,
    _strip_html_tables_and_content,
)

# ── Tuning knobs ─────────────────────────────────────────────────────────────

# Order rules are the expensive half of content faithfulness; a handful of
# well-separated pairs measures reading order without swamping the rule set.
MAX_ORDER_RULES = 20
MIN_ORDER_SENTENCE_LENGTH = 12

MAX_FORMATTING_RULES_PER_KIND = 25
MAX_NEGATIVE_RULES_PER_KIND = 10
MAX_TITLE_RULES = 25
MAX_FORMATTED_SPAN_LENGTH = 160
MIN_DISTINCTIVE_SPAN_LENGTH = 4

# Formatting kinds the evaluator scores as "text styling" (F0.5-weighted, so
# false positives hurt more than misses — hence the negative rules).
STYLING_KINDS = ("bold", "strikeout", "sup", "sub")

_BOLD_PATTERN = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", re.DOTALL)
_ITALIC_PATTERN = re.compile(r"(?<![*\w])\*(?!\s|\*)(.+?)(?<!\s)\*(?!\*)", re.DOTALL)
_STRIKEOUT_PATTERN = re.compile(r"~~(?!\s)(.+?)(?<!\s)~~", re.DOTALL)
_SUP_PATTERN = re.compile(r"<sup>(.+?)</sup>", re.IGNORECASE | re.DOTALL)
_SUB_PATTERN = re.compile(r"<sub>(.+?)</sub>", re.IGNORECASE | re.DOTALL)
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

_FORMATTING_PATTERNS: dict[str, re.Pattern[str]] = {
    "bold": _BOLD_PATTERN,
    "italic": _ITALIC_PATTERN,
    "strikeout": _STRIKEOUT_PATTERN,
    "sup": _SUP_PATTERN,
    "sub": _SUB_PATTERN,
}

_INLINE_MARKUP = re.compile(r"\*\*|~~|[*_`]|</?[a-zA-Z][^>]*>")


@dataclass
class DerivedRules:
    """Rules derived for one document, grouped by evaluation category."""

    by_category: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add(self, category: str, rule: dict[str, Any]) -> None:
        self.by_category.setdefault(category, []).append(rule)

    def count(self, category: str) -> int:
        return len(self.by_category.get(category, []))

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_category.values())


class _RuleIds:
    """Stable, human-readable rule ids scoped to one document."""

    def __init__(self, doc_stem: str) -> None:
        self.doc_stem = doc_stem
        self._counters: Counter[str] = Counter()

    def next(self, rule_type: str) -> str:
        self._counters[rule_type] += 1
        return f"{self.doc_stem}::{rule_type}::{self._counters[rule_type]}"


def _clean_span_text(text: str) -> str:
    """Strip nested markup from a formatted span, leaving the plain query text."""
    cleaned = _INLINE_MARKUP.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def ordered_sentences(markdown: str) -> list[str]:
    """Normalized sentences in document order.

    ``SentenceBagRule`` only exposes an unordered Counter, but order rules need
    the sequence. This mirrors ``_extract_normalized_sentences_static`` step for
    step so the fragments are byte-identical to what evaluation will look for.
    """
    content = _strip_fenced_code_blocks(markdown)
    content = _strip_and_replace_latex(content)
    content = SentenceBagRule._MARKDOWN_IMAGE_PATTERN.sub(" ", content)
    content = _strip_html_tables_and_content(content)
    content = SentenceBagRule._MULTI_DOT_PATTERN.sub(" ", content)
    chunks = SentenceBagRule._SENTENCE_SPLIT_PATTERN.split(content)
    chunks = SentenceBagRule._merge_short_chunks(chunks)

    sentences: list[str] = []
    for chunk in chunks:
        normalized = SentenceBagRule._normalize_sentence_fragment(chunk)
        if normalized:
            sentences.append(normalized)
    return sentences


def derive_text_content_rules(markdown: str, ids: _RuleIds) -> list[dict[str, Any]]:
    """Bag and order rules feeding ``content_faithfulness``."""
    rules: list[dict[str, Any]] = []

    word_bag = WordBagRule._extract_normalized_words_static(markdown, include_table_cells=True)
    sentence_bag = SentenceBagRule._extract_normalized_sentences_static(markdown)
    digit_bag = _extract_digit_counts(markdown, include_table_cells=True)

    if word_bag:
        bag = dict(word_bag)
        for rule_type in (
            "missing_word_percent",
            "unexpected_word_percent",
            "too_many_word_occurence_percent",
        ):
            rules.append({"type": rule_type, "id": ids.next(rule_type), "bag_of_word": bag})

    if sentence_bag:
        bag = dict(sentence_bag)
        for rule_type in (
            "missing_sentence_percent",
            "unexpected_sentence_percent",
            "too_many_sentence_occurence_percent",
        ):
            rules.append({"type": rule_type, "id": ids.next(rule_type), "bag_of_sentence": bag})

    if digit_bag:
        rules.append(
            {
                "type": "bag_of_digit_percent",
                "id": ids.next("bag_of_digit_percent"),
                "bag_of_digit": dict(digit_bag),
            }
        )

    rules.extend(_derive_order_rules(markdown, sentence_bag, ids))
    return rules


def _derive_order_rules(
    markdown: str,
    sentence_bag: Counter[str],
    ids: _RuleIds,
) -> list[dict[str, Any]]:
    """Reading-order rules from sentences that appear exactly once.

    A sentence occurring more than once has no single position, so an
    order assertion about it would be ambiguous and unfairly scored.
    """
    sequence = [
        s for s in ordered_sentences(markdown) if sentence_bag.get(s, 0) == 1 and len(s) >= MIN_ORDER_SENTENCE_LENGTH
    ]
    if len(sequence) < 2:
        return []

    # Spread pairs across the document rather than clustering at the top.
    pair_count = min(MAX_ORDER_RULES, len(sequence) - 1)
    stride = max(1, (len(sequence) - 1) // pair_count)

    rules: list[dict[str, Any]] = []
    for i in range(0, len(sequence) - 1, stride):
        if len(rules) >= MAX_ORDER_RULES:
            break
        rules.append(
            {
                "type": "order",
                "id": ids.next("order"),
                "before": sequence[i],
                "after": sequence[i + 1],
            }
        )
    return rules


def _find_formatted_spans(markdown: str) -> dict[str, list[str]]:
    """Plain text of every formatted span, keyed by formatting kind."""
    spans: dict[str, list[str]] = {}
    for kind, pattern in _FORMATTING_PATTERNS.items():
        found: list[str] = []
        seen: set[str] = set()
        for match in pattern.finditer(markdown):
            text = _clean_span_text(match.group(1))
            if not text or len(text) > MAX_FORMATTED_SPAN_LENGTH:
                continue
            if text in seen:
                continue
            seen.add(text)
            found.append(text)
        if found:
            spans[kind] = found
    return spans


def _derive_negative_formatting_rules(
    markdown: str,
    spans: dict[str, list[str]],
    ids: _RuleIds,
) -> list[dict[str, Any]]:
    """Assertions that unformatted text must not come back formatted.

    Text styling is scored with an F0.5 mean, so a parser that bolds half the
    page is penalised harder than one that misses bold. Those false positives
    are only detectable with negative rules.
    """
    # Overlap is only checked against spans long enough to be distinctive. A
    # one-character span like the "a" in "clause 4<sup>a</sup>" is a substring
    # of nearly every line, and using it would reject the whole document.
    formatted_text = {text for texts in spans.values() for text in texts if len(text) >= MIN_DISTINCTIVE_SPAN_LENGTH}
    plain_lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("<"):
            continue
        if _INLINE_MARKUP.search(line):
            continue
        if len(line) < MIN_ORDER_SENTENCE_LENGTH or len(line) > MAX_FORMATTED_SPAN_LENGTH:
            continue
        if any(line in text or text in line for text in formatted_text):
            continue
        plain_lines.append(line)

    rules: list[dict[str, Any]] = []
    for kind in STYLING_KINDS:
        # Only assert absence of a style the document actually uses somewhere;
        # otherwise the rule tests nothing the evaluator weighs.
        if kind not in spans:
            continue
        for line in plain_lines[:MAX_NEGATIVE_RULES_PER_KIND]:
            rule_type = f"is_not_{kind}"
            rules.append({"type": rule_type, "id": ids.next(rule_type), "text": line})
    return rules


def _derive_title_rules(markdown: str, ids: _RuleIds) -> list[dict[str, Any]]:
    """Heading rules plus one nested-hierarchy rule."""
    headings = [(len(m.group(1)), _clean_span_text(m.group(2))) for m in _HEADING_PATTERN.finditer(markdown)]
    headings = [(level, text) for level, text in headings if text]
    if not headings:
        return []

    rules: list[dict[str, Any]] = []
    for level, text in headings[:MAX_TITLE_RULES]:
        rules.append({"type": "is_title", "id": ids.next("is_title"), "text": text, "level": level})

    hierarchy = _build_title_hierarchy(headings)
    if hierarchy:
        rules.append(
            {
                "type": "title_hierarchy_percent",
                "id": ids.next("title_hierarchy_percent"),
                "title_hierarchy": hierarchy,
            }
        )
    return rules


def _build_title_hierarchy(headings: list[tuple[int, str]]) -> dict[str, Any]:
    """Nest headings by level into the map ``title_hierarchy_percent`` expects."""
    root: dict[str, Any] = {}
    # Stack of (level, children-dict-of-that-heading).
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]

    for level, text in headings:
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        parent_children = stack[-1][1]
        children: dict[str, Any] = {}
        parent_children[text] = children
        stack.append((level, children))
    return root


def derive_text_formatting_rules(markdown: str, ids: _RuleIds) -> list[dict[str, Any]]:
    """Styling and title rules feeding ``semantic_formatting``."""
    rules: list[dict[str, Any]] = []
    spans = _find_formatted_spans(markdown)

    for kind, texts in spans.items():
        rule_type = f"is_{kind}"
        for text in texts[:MAX_FORMATTING_RULES_PER_KIND]:
            rules.append({"type": rule_type, "id": ids.next(rule_type), "text": text})

    rules.extend(_derive_negative_formatting_rules(markdown, spans, ids))
    rules.extend(_derive_title_rules(markdown, ids))
    return rules


def derive_chart_rules(charts: list[dict[str, Any]], ids: _RuleIds) -> list[dict[str, Any]]:
    """``chart_data_point`` rules from the model's chart reading."""
    rules: list[dict[str, Any]] = []
    for chart in charts:
        title = str(chart.get("title") or "").strip()
        for point in chart.get("points") or []:
            if not isinstance(point, dict):
                continue
            value = point.get("value")
            labels = point.get("labels")
            if value is None or not isinstance(labels, list) or not labels:
                continue
            clean_labels = [str(label).strip() for label in labels if str(label).strip()]
            if not clean_labels:
                continue
            if title and title not in clean_labels:
                clean_labels = [title, *clean_labels]
            rules.append(
                {
                    "type": "chart_data_point",
                    "id": ids.next("chart_data_point"),
                    "value": value,
                    "labels": clean_labels,
                    "normalize_numbers": True,
                }
            )
    return rules


def derive_rules(
    doc_stem: str,
    markdown: str,
    charts: list[dict[str, Any]],
    categories: list[str],
) -> DerivedRules:
    """Derive every requested category's rules from one document's reference.

    :param doc_stem: Document filename stem, used to scope rule ids.
    :param markdown: Reference transcription for the whole document.
    :param charts: Chart readings, if the chart category was requested.
    :param categories: Evaluation categories to derive.
    """
    ids = _RuleIds(doc_stem)
    derived = DerivedRules()

    if "text_content" in categories:
        for rule in derive_text_content_rules(markdown, ids):
            derived.add("text_content", rule)

    if "text_formatting" in categories:
        for rule in derive_text_formatting_rules(markdown, ids):
            derived.add("text_formatting", rule)

    if "chart" in categories:
        for rule in derive_chart_rules(charts, ids):
            derived.add("chart", rule)

    # The `table` category is scored from the reference markdown itself
    # (GriTS / TEDS / TableRecordMatch), so it contributes no rules. Its entry
    # is created by the emitter as an expected_markdown pointer row.
    if "table" in categories:
        derived.by_category.setdefault("table", [])

    return derived
