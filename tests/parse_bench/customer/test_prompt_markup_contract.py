"""Guard the contract between the transcription prompt and the evaluator.

The ground-truth prompt tells the model which markup to emit; ``derive.py``
searches for exactly that markup; ``rules_formatting.FormattingRule`` matches
it again at scoring time. If any of the three drifts, formatting ground truth
silently stops being generated — every parser scores zero on semantic
formatting and nobody can tell why. These tests fail loudly instead.
"""

from __future__ import annotations

from parse_bench.customer.groundtruth.derive import (
    STYLING_KINDS,
    _find_formatted_spans,
)
from parse_bench.customer.groundtruth.prompts import (
    MARKUP_CONTRACT,
    TRANSCRIPTION_SYSTEM_PROMPT,
)
from parse_bench.evaluation.metrics.parse.rules_formatting import FormattingRule

# One example of each markup form the prompt promises to emit.
_MARKUP_SAMPLES: dict[str, str] = {
    "bold": "This is **a bold span** in a line.",
    "italic": "This is *an italic span* in a line.",
    "strikeout": "This is ~~a struck span~~ in a line.",
    "sup": "This is a footnote<sup>ref</sup> marker.",
    "sub": "The formula H<sub>2</sub>O appears here.",
}


# The exact syntax the prompt must instruct the model to emit, per kind.
_CONTRACT_SYNTAX: dict[str, str] = {
    "bold": "**text**",
    "italic": "*text*",
    "strikeout": "~~text~~",
    "sup": "<sup>text</sup>",
    "sub": "<sub>text</sub>",
}


class TestPromptDeclaresWhatWeParse:
    def test_every_derived_kind_is_documented_in_the_contract(self) -> None:
        for kind, syntax in _CONTRACT_SYNTAX.items():
            assert syntax in MARKUP_CONTRACT, f"prompt never asks the model to emit {kind} as {syntax}"

    def test_contract_is_embedded_in_the_system_prompt(self) -> None:
        assert MARKUP_CONTRACT in TRANSCRIPTION_SYSTEM_PROMPT

    def test_contract_names_html_tables(self) -> None:
        # Table metrics only fire on HTML tables; markdown pipes score nothing.
        assert "<table>" in MARKUP_CONTRACT
        assert "colspan" in MARKUP_CONTRACT


class TestDerivationFindsPromptedMarkup:
    def test_each_markup_form_is_extracted(self) -> None:
        for kind, sample in _MARKUP_SAMPLES.items():
            spans = _find_formatted_spans(sample)
            assert kind in spans, f"derive.py cannot find {kind} markup the prompt asks for"

    def test_bold_is_not_mistaken_for_italic(self) -> None:
        spans = _find_formatted_spans("A **bold** run and an *italic* run.")
        assert spans.get("bold") == ["bold"]
        assert spans.get("italic") == ["italic"]


class TestEvaluatorMatchesDerivedRules:
    def test_positive_rules_pass_against_their_own_markup(self) -> None:
        for kind, sample in _MARKUP_SAMPLES.items():
            spans = _find_formatted_spans(sample)
            text = spans[kind][0]
            rule = FormattingRule({"type": f"is_{kind}", "text": text})
            passed, message = rule.run(sample)[:2]
            assert passed, f"is_{kind} rule for {text!r} does not match its own source: {message}"

    def test_negative_rules_fail_when_the_style_is_present(self) -> None:
        for kind in STYLING_KINDS:
            sample = _MARKUP_SAMPLES[kind]
            text = _find_formatted_spans(sample)[kind][0]
            rule = FormattingRule({"type": f"is_not_{kind}", "text": text})
            passed = rule.run(sample)[0]
            assert not passed, f"is_not_{kind} passed on text that is {kind}"

    def test_negative_rules_pass_on_plain_text(self) -> None:
        plain = "A line with no formatting at all in it."
        for kind in STYLING_KINDS:
            rule = FormattingRule({"type": f"is_not_{kind}", "text": "no formatting at all"})
            assert rule.run(plain)[0], f"is_not_{kind} failed on genuinely plain text"

    def test_styling_kinds_are_the_ones_the_evaluator_weighs(self) -> None:
        # evaluators/parse.py folds exactly these pairs into
        # normalized_text_styling; generating negatives for anything else
        # would produce rules that score nothing.
        assert set(STYLING_KINDS) == {"bold", "strikeout", "sup", "sub"}
