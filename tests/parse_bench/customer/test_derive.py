"""Rule derivation from a reference transcription.

The load-bearing property is that derived rules are *satisfiable by the
reference itself*: a parser that reproduces the reference exactly must score
1.0. If that fails, every pipeline is scored against an impossible target and
the whole comparison is worthless.
"""

from __future__ import annotations

import pytest

from parse_bench.customer.groundtruth.derive import (
    _build_title_hierarchy,
    _RuleIds,
    derive_chart_rules,
    derive_rules,
    derive_text_content_rules,
    derive_text_formatting_rules,
    ordered_sentences,
)
from parse_bench.evaluation.metrics.parse.rules_bag import SentenceBagRule

REFERENCE = """\
# Annual Policy Summary

The insured party is Acme Insurance Limited of Dublin.

Coverage begins on 1 January 2026 and ends on 31 December 2026.

## Premiums

The **total premium due** is 1200 euro per annum, payable monthly.

The previous rate of ~~1450 euro~~ no longer applies.

Claims must be submitted within thirty days of the incident occurring.

### Exclusions

Flood damage is excluded under clause 4<sup>a</sup> of this agreement.
"""


def _ids() -> _RuleIds:
    return _RuleIds("doc")


class TestOrderedSentences:
    def test_matches_the_evaluator_bag_exactly(self) -> None:
        # Ordering is reimplemented here because the evaluator only exposes a
        # Counter; the fragments themselves must stay identical or order rules
        # would reference text the evaluator never produces.
        sequence = ordered_sentences(REFERENCE)
        expected = SentenceBagRule._extract_normalized_sentences_static(REFERENCE)
        assert sorted(sequence) == sorted(expected.elements())

    def test_preserves_document_order(self) -> None:
        sequence = ordered_sentences(REFERENCE)
        # Fragments come back normalized (lowercased) by the evaluator's tokenizer.
        insured = next(i for i, s in enumerate(sequence) if "insured party" in s)
        claims = next(i for i, s in enumerate(sequence) if "claims must be submitted" in s)
        assert insured < claims


class TestTextContentRules:
    def test_emits_the_rule_types_content_faithfulness_scores(self) -> None:
        rules = derive_text_content_rules(REFERENCE, _ids())
        types = {r["type"] for r in rules}
        # These are exactly the types evaluators/parse.py folds into
        # normalized_text_correctness and normalized_order.
        assert {
            "missing_word_percent",
            "unexpected_word_percent",
            "too_many_word_occurence_percent",
            "missing_sentence_percent",
            "unexpected_sentence_percent",
            "too_many_sentence_occurence_percent",
            "bag_of_digit_percent",
            "order",
        } <= types

    def test_word_bag_is_populated_with_counts(self) -> None:
        rules = derive_text_content_rules(REFERENCE, _ids())
        bag = next(r for r in rules if r["type"] == "missing_word_percent")["bag_of_word"]
        assert bag
        assert all(isinstance(count, int) and count > 0 for count in bag.values())

    def test_order_rules_point_forward(self) -> None:
        rules = derive_text_content_rules(REFERENCE, _ids())
        sequence = ordered_sentences(REFERENCE)
        for rule in (r for r in rules if r["type"] == "order"):
            assert sequence.index(rule["before"]) < sequence.index(rule["after"])

    def test_empty_document_yields_no_rules(self) -> None:
        assert derive_text_content_rules("", _ids()) == []

    def test_rule_ids_are_unique(self) -> None:
        rules = derive_text_content_rules(REFERENCE, _ids())
        ids = [r["id"] for r in rules]
        assert len(ids) == len(set(ids))


class TestFormattingRules:
    def test_finds_bold_strikeout_and_superscript(self) -> None:
        rules = derive_text_formatting_rules(REFERENCE, _ids())
        by_type: dict[str, list[str]] = {}
        for rule in rules:
            by_type.setdefault(rule["type"], []).append(rule.get("text", ""))

        assert "total premium due" in [t.lower() for t in by_type.get("is_bold", [])]
        assert "1450 euro" in by_type.get("is_strikeout", [])
        assert "a" in by_type.get("is_sup", [])

    def test_titles_carry_their_level(self) -> None:
        rules = derive_text_formatting_rules(REFERENCE, _ids())
        titles = {r["text"]: r["level"] for r in rules if r["type"] == "is_title"}
        assert titles["Annual Policy Summary"] == 1
        assert titles["Premiums"] == 2
        assert titles["Exclusions"] == 3

    def test_negative_rules_only_for_styles_the_document_uses(self) -> None:
        # is_not_sub would test nothing: the evaluator only weighs a negative
        # alongside its positive.
        rules = derive_text_formatting_rules(REFERENCE, _ids())
        types = {r["type"] for r in rules}
        assert "is_not_bold" in types
        assert "is_not_sub" not in types

    def test_negative_rule_text_is_genuinely_unformatted(self) -> None:
        rules = derive_text_formatting_rules(REFERENCE, _ids())
        for rule in (r for r in rules if r["type"].startswith("is_not_")):
            assert "**" not in rule["text"]
            assert "~~" not in rule["text"]

    def test_plain_document_produces_no_styling_rules(self) -> None:
        rules = derive_text_formatting_rules("Just a line of plain text here.\n", _ids())
        assert all(not r["type"].startswith("is_bold") for r in rules)


class TestTitleHierarchy:
    def test_nests_by_heading_level(self) -> None:
        hierarchy = _build_title_hierarchy([(1, "A"), (2, "B"), (3, "C"), (2, "D")])
        assert hierarchy == {"A": {"B": {"C": {}}, "D": {}}}

    def test_sibling_top_level_headings(self) -> None:
        assert _build_title_hierarchy([(1, "A"), (1, "B")]) == {"A": {}, "B": {}}

    def test_deeper_first_heading_does_not_crash(self) -> None:
        assert _build_title_hierarchy([(3, "Deep"), (1, "Top")]) == {"Deep": {}, "Top": {}}


class TestChartRules:
    def test_builds_data_point_rules(self) -> None:
        charts = [{"title": "Revenue", "points": [{"value": 12.4, "labels": ["Q3 2025"]}]}]
        rules = derive_chart_rules(charts, _ids())
        assert len(rules) == 1
        assert rules[0]["value"] == 12.4
        assert "Revenue" in rules[0]["labels"]

    def test_skips_points_without_a_value_or_labels(self) -> None:
        charts = [
            {
                "title": "Revenue",
                "points": [
                    {"value": None, "labels": ["Q1"]},
                    {"value": 5, "labels": []},
                    {"value": 7, "labels": ["Q2"]},
                ],
            }
        ]
        assert len(derive_chart_rules(charts, _ids())) == 1

    def test_no_charts_is_not_an_error(self) -> None:
        assert derive_chart_rules([], _ids()) == []


class TestDeriveRules:
    def test_respects_the_requested_categories(self) -> None:
        derived = derive_rules("doc", REFERENCE, [], ["text_content"])
        assert set(derived.by_category) == {"text_content"}

    def test_table_category_is_scored_from_markdown_not_rules(self) -> None:
        # Table metrics (GriTS/TEDS/TRM) read expected_markdown directly, so
        # the category is present but carries no rules.
        derived = derive_rules("doc", REFERENCE, [], ["table"])
        assert derived.by_category["table"] == []

    def test_derived_rules_pass_the_evaluator_schema(self) -> None:
        from parse_bench.test_cases.parse_rule_schemas import coerce_parse_rule_list

        derived = derive_rules("doc", REFERENCE, [], ["text_content", "text_formatting"])
        for rules in derived.by_category.values():
            if rules:
                coerce_parse_rule_list([dict(r) for r in rules])


class TestReferenceScoresPerfectly:
    """The reference must be able to score 1.0 against its own rules."""

    @pytest.mark.parametrize(
        "rule_type",
        [
            "missing_word_percent",
            "unexpected_word_percent",
            "missing_sentence_percent",
            "unexpected_sentence_percent",
        ],
    )
    def test_bag_rules_are_satisfied_by_the_reference(self, rule_type: str) -> None:
        from parse_bench.evaluation.metrics.parse.rules_bag import (
            MissingSentencePercentRule,
            MissingWordPercentRule,
            UnexpectedSentencePercentRule,
            UnexpectedWordPercentRule,
        )

        implementations = {
            "missing_word_percent": MissingWordPercentRule,
            "unexpected_word_percent": UnexpectedWordPercentRule,
            "missing_sentence_percent": MissingSentencePercentRule,
            "unexpected_sentence_percent": UnexpectedSentencePercentRule,
        }
        rules = derive_text_content_rules(REFERENCE, _ids())
        payload = next(r for r in rules if r["type"] == rule_type)
        rule = implementations[rule_type](payload)
        _, _, score = rule.run(REFERENCE)
        assert score == pytest.approx(1.0), f"{rule_type} cannot be satisfied by its own reference"

    def test_order_rules_are_satisfied_by_the_reference(self) -> None:
        from parse_bench.evaluation.metrics.parse.rules_text import TextOrderRule

        rules = derive_text_content_rules(REFERENCE, _ids())
        order_rules = [r for r in rules if r["type"] == "order"]
        assert order_rules
        for payload in order_rules:
            passed, message = TextOrderRule(payload).run(REFERENCE)[:2]
            assert passed, message
