# Customer evaluation workflow

Run ParseBench on documents that cannot leave the machine they live on.

The public benchmark answers "which parser is best on ParseBench?". This
answers "which parser is best on *my* documents?" — without the documents
going anywhere, and without anyone having to take a vendor's word for the
result.

- **Runbook for running this with someone** → [`docs/customer_eval.md`](../../../docs/customer_eval.md)
- **This document** → how the machinery works and why it is built this way

---

## The five commands

```bash
parse-bench customer init ./acme --pipelines llamaparse_agentic,reducto,extend_parse
#   → scaffolds a project, writes a .env listing exactly the keys those pipelines need

parse-bench customer ingest ./acme
#   → stages documents from docs/ into the dataset layout

parse-bench customer groundtruth ./acme        # --dry_run first for a cost estimate
#   → transcribes each page, derives evaluation rules from the transcription

parse-bench customer run ./acme
#   → runs every configured pipeline; one failing does not stop the rest

parse-bench customer report ./acme
#   → paired statistical comparison, as HTML + markdown + JSON
```

`parse-bench customer status ./acme` prints how far along a project is and
what to do next. `parse-bench customer keys ./acme` shows which credentials
are still missing.

---

## How it works

```
docs/                     ingest        data/                    run              output/
┌──────────────┐                   ┌──────────────────┐                   ┌────────────────┐
│ table/       │  ──────────────▶  │ pdfs/{group}/    │  ──────────────▶  │ <pipeline>/    │
│ chart/       │   copy + cap      │                  │   stock inference │   *.result.json│
│ text/        │   long documents  │ {category}.jsonl │   + evaluation    │   _evaluation_ │
└──────────────┘                   │ expected_        │                   │    report.json │
                                   │   markdown.json  │                   └────────┬───────┘
       groundtruth                 │ _groundtruth/    │                            │
┌──────────────────┐               │   *.md  ◀── review these                      │ report
│ render page      │  ──────────▶  └──────────────────┘                            ▼
│ → vision model   │   transcription        ▲                            ┌────────────────┐
│ → reference md   │                        │                            │ reports/       │
└──────────────────┘                        │ derived deterministically  │  comparison_   │
                                            └────────────────────────────│   report.html  │
                                                                         └────────────────┘
```

Everything under `data/` is the **stock ParseBench dataset format**. It loads
through `parse_bench.test_cases.loader` with no special-casing, which is what
keeps a customer's numbers comparable to the public benchmark rather than a
private fork of it.

---

## The central design decision: the model transcribes, ParseBench derives

The obvious way to bootstrap ground truth is to ask a model to write
evaluation rules. That is the wrong shape, for two reasons: a model asked to
author rules will invent assertions the document does not support, and it puts
a model in the position of deciding what "correct" means.

Instead the model has exactly one job — **transcribe the page** — and the
rules are derived from that transcription by deterministic code:

| Dimension | What is derived | Feeds |
|---|---|---|
| Content faithfulness | word / sentence / digit bags, reading-order pairs | `content_faithfulness` |
| Semantic formatting | bold, strikethrough, super/subscript spans, headings | `semantic_formatting` |
| Tables | the reference markdown itself — no rules needed | GriTS + TableRecordMatch |
| Charts | data points read from each chart | `ChartDataPointMatch` |

Two properties fall out of this, and both matter when someone challenges the
result:

**Scoring stays deterministic.** ParseBench does not use LLM-as-a-judge, and
this workflow does not change that. A model wrote a reference *document*; no
model scores anything.

**The bags are built by the code that reads them.** `derive.py` calls
`rules_bag.WordBagRule._extract_normalized_words_static` and
`SentenceBagRule._extract_normalized_sentences_static` — the evaluator's own
tokenizers. Expected and actual text are therefore normalized identically, so
a rule can never fail because the two sides disagreed about what a word is.
`tests/parse_bench/customer/test_derive.py` asserts the load-bearing property:
**the reference scores 1.0 against its own derived rules.** If that ever
breaks, every pipeline is being scored against an impossible target.

### Which rule types, and why those

This is worth stating because it is easy to get wrong. `content_faithfulness`
is computed in `evaluators/parse.py` from `normalized_text_correctness` (the
`*_percent` bag rules, `extra_content`, `bag_of_digit_percent`) plus
`normalized_order` (`order` rules) — **`present` rules do not contribute to
it.** A bootstrap that emitted `present` rules would produce a dataset that
looks well-populated and moves no metric the report keys off.

All of those rule types are mechanically derivable from an accurate
transcription, which is why the prompt never mentions ParseBench, rule types,
or JSONL. What it does specify carefully is the markup contract — the
evaluator can only match formatting it recognises, so the prompt, `derive.py`,
and `rules_formatting.FormattingRule` have to agree on `**bold**`, `~~strike~~`,
`<sup>`, `<sub>`, and HTML tables with `colspan`.
`test_prompt_markup_contract.py` fails loudly if any of the three drifts;
without it, formatting ground truth would silently stop being generated and
every parser would score zero on semantic formatting with no visible cause.

### What is deliberately not bootstrapped

Visual grounding (`layout`). Bounding boxes from a VLM are not accurate enough
to defend in front of a customer, so that dimension needs hand-drawn boxes via
[`apps/annotator`](../../../apps/annotator) or the customer's own coordinates.

---

## Provenance and honesty

Bootstrapped ground truth is a starting point, not gospel, and the tooling is
built so nobody can mistake one for the other:

- every generated rule is written `verified: false` and tagged `bootstrap`
- the report header states the human-verified percentage
- reference transcriptions land in `data/_groundtruth/**.md` as plain markdown

That last one is the most useful thing in a live conversation: opening a
reference next to the source PDF makes the ground truth *inspectable* rather
than asserted. Corrections take effect on the next `run` — rules are derived
from the files on disk, so nothing needs regenerating.

Documents longer than `max_pages_per_doc` are truncated **before** parsing, so
the evaluated document and the ground truth cover the same pages. Without
that, a parser would be penalised for faithfully transcribing pages the
reference never described.

---

## Statistics

Documents are the unit of analysis and every pipeline sees the same documents,
so the comparison is paired throughout — which is what controls for the fact
that some documents are simply harder than others.

Per challenger, against the baseline:

- **mean difference + 95% CI** — percentile bootstrap, 10,000 resamples, fixed
  seed, so re-running the report gives identical numbers
- **p-value** — two-sided Wilcoxon signed-rank, Holm-corrected across the
  challengers in that dimension. Non-parametric because parse scores are
  bounded, skewed, and tie-heavy, which breaks the t-test's assumptions
- **win / loss / tie** — the honest picture when a mean hides a bimodal split
- **effect size** — Cohen's *dz*

Dimensions with fewer than 10 documents are flagged **underpowered**: the
numbers are shown, but they should not settle an argument.

One consequence worth knowing before a demo: comparing five pipelines means
four tests, and Holm correction costs real power. A gap that clears p<0.05 on
its own can land at p≈0.09 once corrected. That is the correction working, not
a bug — but it means you should decide which comparison you care about
*before* running, and keep the challenger set small. Both raw and adjusted
p-values are in `reports/comparison_report.json`.

---

## Module map

| Module | Responsibility |
|---|---|
| `cli.py` | The seven commands; project loading and `.env` handling |
| `project.py` | Directory layout, config schema, API-key discovery |
| `ingest.py` | Staging documents, page capping, duplicate detection |
| `groundtruth/render.py` | Page → PNG (PyMuPDF), PDF truncation |
| `groundtruth/client.py` | OpenAI-compatible vision client (httpx only) |
| `groundtruth/prompts.py` | Transcription + chart prompts, markup contract |
| `groundtruth/derive.py` | Reference markdown → evaluation rules |
| `groundtruth/emit.py` | Rules → ParseBench JSONL + `expected_markdown.json` |
| `groundtruth/generate.py` | Orchestration, resume, cost estimation |
| `comparison/scores.py` | Per-document scores out of evaluation output |
| `comparison/stats.py` | Bootstrap CIs, Wilcoxon, Holm, effect sizes |
| `comparison/report.py` | HTML / markdown / JSON report |

API-key discovery is worth one note: it finds the variables a pipeline reads
by scanning provider source for `os.getenv(...)`, rather than importing the
provider. Provider modules are imported lazily and skipped when their vendor
SDK is absent, so importing would produce an empty key template for a customer
who has not yet run `uv sync --extra runners` — precisely when they most need
to know which keys to go and fetch.

---

## Worked example

Fully reproducible, no API keys, nothing leaves the machine. It uses local
parsers (`pymupdf_text`, `pypdf_baseline`, `pymupdf_html`) so the only thing
being demonstrated is the machinery.

```bash
parse-bench customer init ./demo \
    --name "Demo" --pipelines pymupdf_text,pypdf_baseline,pymupdf_html

# drop a few dozen PDFs into ./demo/docs/text/
parse-bench customer ingest ./demo
parse-bench customer groundtruth ./demo      # needs OPENROUTER_API_KEY
parse-bench customer run ./demo
parse-bench customer report ./demo
```

On a 26-document set the report reads like this:

```
Energy (rule_pass_rate)
  * pymupdf_text      44.1  (n=26)
    pypdf_baseline    43.7  (n=26)
    pymupdf_html      13.9  (n=26)

  pypdf_baseline vs baseline — No significant difference (0.3 point gap, p=1.000, n=26).
  pymupdf_html   vs baseline — pymupdf_text is better by 30.2 points
                               (95% CI -41.8 to -18.9, p=0.0004, n=26).
```

That is the behaviour to look for: a 30-point gap comes back significant with
a tight interval, and a 0.4-point gap is called inconclusive rather than
dressed up as a win. Reporting the second one as a narrow victory is how
technical credibility gets spent for nothing.

Swapping in hosted parsers is a one-line change to `pipelines` in
`parsebench_customer.json` plus the relevant key in `.env`.

---

## Bring your own ground truth

If the customer already has labels, skip `groundtruth` entirely — no document
then reaches any model except the parsers under evaluation. Both dataset
formats are accepted:

**JSONL** (`data/{category}.jsonl`, plus `data/expected_markdown.json`):

```json
{"pdf": "pdfs/text/policy_01.pdf", "category": "text_content",
 "id": "policy_01::order::1", "type": "order", "verified": true,
 "rule": {"before": "the insured party is acme limited",
          "after": "coverage begins on 1 january 2026"}}
```

**Sidecar** (`data/<group>/<name>.test.json` next to each PDF) — often what
an existing internal benchmark already looks like:

```json
{"expected_markdown": "...", "tags": ["energy"],
 "test_rules": [{"type": "present", "text": "...", "max_l_dist": 5}]}
```

`apps/annotator/rule_definitions.json` documents every rule type and its
fields. A hybrid is frequently the strongest option: bootstrap the bulk, have
the customer hand-verify a held-out subset, and report both. Agreement between
the two earns the bootstrapped ground truth its credibility on their data
specifically.

### A note on reusing an internal benchmark

If you point this at ground truth that was tuned while looking at one parser's
output — thresholds loosened until near-misses passed, anchors chosen because
a particular parser preserves them — that parser has a head start, and a
prospect's data-science team will find it. The failure mode is subtle because
such adjustments are usually made in good faith as fairness fixes.

It is worth checking rather than assuming, and the check is cheap: split the
documents into those a recalibration pass touched and those it did not, and
confirm the ranking survives on the untouched subset. If it does not, the
ranking was an artifact. If it does, you have an answer that holds up under
exactly the question a sceptic will ask.

---

## Tests

```bash
uv run pytest tests/parse_bench/customer/ -q
```

The ones that carry weight:

| Test | Guards |
|---|---|
| `test_derive.py::TestReferenceScoresPerfectly` | Derived rules are satisfiable by their own reference |
| `test_emit.py::TestLoaderRoundTrip` | Generated data loads through the stock loader unchanged |
| `test_prompt_markup_contract.py` | Prompt, derivation, and evaluator agree on markup |
| `test_scores.py::TestSingleDimensionLayout` | Single-dimension runs are not silently invisible |
| `test_stats.py` | Bootstrap determinism, Holm monotonicity, underpowered flagging |

---

## Known limitations

- **Visual grounding is not bootstrapped.** Needs the annotator or customer boxes.
- **Charts depend on the model reading values correctly.** The prompt forbids
  estimating from pixel heights, but chart ground truth deserves more human
  review than text does.
- **Ground-truth generation is expensive** — budget around $0.70 per page. It
  is the cost of manufacturing reference data, not of parsing; production
  parsing is orders of magnitude cheaper, which is itself worth telling a
  prospect.
- **`docx` documents are staged but not rendered** for ground-truth
  generation; convert to PDF first.
