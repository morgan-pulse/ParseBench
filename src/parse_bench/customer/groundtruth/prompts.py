"""Prompts for ground-truth transcription.

There is exactly one perception task — transcribe the page — plus an optional
chart-reading task. Evaluation rules are derived from the transcription in
``derive.py``, so these prompts never mention rule types or JSONL. What they do
specify carefully is the *markup contract*: the derivation step can only find
formatting that the transcription actually marks up, and the evaluator can only
match markup it recognises (see ``rules_formatting.FormattingRule``).
"""

from __future__ import annotations

# Markup the evaluator recognises. Kept in sync with
# evaluation/metrics/parse/rules_formatting.py — see
# tests/parse_bench/customer/test_prompt_markup_contract.py.
MARKUP_CONTRACT = """\
- Bold: **text**
- Italic: *text*
- Strikethrough: ~~text~~
- Superscript: <sup>text</sup>
- Subscript: <sub>text</sub>
- Headings: # / ## / ### by visual hierarchy, most prominent first
- Tables: HTML <table> with <tr>/<td>, using colspan and rowspan for merged cells
- Formulas: $...$ for inline, $$...$$ for display
- Code: fenced blocks with the language tag"""

TRANSCRIPTION_SYSTEM_PROMPT = f"""\
You are producing reference ground truth for a document-parsing benchmark. \
Another system will be scored against your transcription, so accuracy matters \
more than tidiness. A word you invent becomes a word every parser is punished \
for omitting; a word you skip becomes a word every parser is punished for \
including.

Transcribe the page image into markdown, in natural reading order.

Rules:
1. Transcribe every visible text element: body text, headers, footers, page \
numbers, captions, footnotes, labels, stamps, and handwriting. Do not \
summarise, correct, reorder, or translate anything.
2. Reproduce text exactly as printed, including typos, odd spacing inside \
identifiers, and original casing.
3. Use this markup, and only this markup:
{MARKUP_CONTRACT}
4. Mark formatting only when it is visually unambiguous. Bold that you are not \
sure about should be left unmarked — a wrong positive is worse than a miss.
5. Tables must be HTML, never markdown pipes. Merged cells must carry colspan \
or rowspan. Preserve the header structure exactly, including stacked or \
hierarchical headers.
6. For images, charts, and diagrams, transcribe any text they contain \
(title, axis labels, legend entries, data labels). Do not describe the artwork.
7. If the page is blank or has no legible text, return an empty markdown string.

Return JSON:
{{"markdown": "<the transcription>", "notes": "<optional caveats, may be empty>"}}"""

TRANSCRIPTION_USER_PROMPT = """\
Transcribe page {page} of {total} of the document "{filename}"."""


CHART_SYSTEM_PROMPT = """\
You are producing reference ground truth for a document-parsing benchmark. \
Read every chart on the page and report its underlying data points.

For each data point, give the value and the labels that identify it. Labels are \
the axis category and series name as printed on the chart — for example \
["Revenue", "Q3 2025"] for the Q3 2025 bar of the Revenue series.

Rules:
1. Report only values you can read directly from a data label, an axis \
gridline the point clearly sits on, or a legend-linked value. Do not estimate \
from pixel heights.
2. Use the numeric value as printed. If the axis is in millions and the label \
reads "12.4", report 12.4, and put the unit in the labels.
3. Labels must match the chart's own wording, exactly as printed.
4. If the page has no chart, or no value can be read with confidence, return an \
empty list. An empty list is a correct answer.

Return JSON:
{"charts": [{"title": "<chart title>", "points": [{"value": <number or string>, \
"labels": ["<label>", "..."]}]}]}"""

CHART_USER_PROMPT = """\
Read the charts on page {page} of {total} of the document "{filename}"."""


def transcription_user_prompt(filename: str, page: int, total: int) -> str:
    return TRANSCRIPTION_USER_PROMPT.format(filename=filename, page=page, total=total)


def chart_user_prompt(filename: str, page: int, total: int) -> str:
    return CHART_USER_PROMPT.format(filename=filename, page=page, total=total)
