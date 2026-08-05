# Customer evaluations

Running ParseBench on a customer's own documents, inside their own environment,
without them sending you anything.

This exists because of a specific failure mode: a prospect won't share their
documents, so the evaluation happens behind closed doors with no ground truth —
or worse, with a competitor's output used *as* ground truth. That comparison
cannot be won, because it defines the competitor as correct by construction.
Handing the prospect a real harness replaces an unwinnable argument with a
measurable one.

## What the customer runs

Five commands, start to finish:

```bash
uv run parse-bench customer init ./acme --name "Acme" --pipelines llamaparse_agentic,extend_parse
uv run parse-bench customer ingest ./acme
uv run parse-bench customer groundtruth ./acme
uv run parse-bench customer run ./acme
uv run parse-bench customer report ./acme
```

`parse-bench customer status ./acme` prints where they are and what to do next,
which is the command to reach for when a call goes sideways.

Everything lives in one directory and nothing leaves the machine except calls to
the parsing APIs the customer chose to evaluate, plus — during `groundtruth`
only — the ground-truth model. If they already have labels, that step is skipped
and no document reaches any model but the ones under evaluation.

## The project directory

```
acme/
  parsebench_customer.json   # pipelines, dimensions, ground-truth settings
  .env                       # only the keys the chosen pipelines need
  docs/                      # customer drops documents here
    table/  chart/  text/    # optional sorting; loose files default to text
  data/                      # generated dataset, in stock ParseBench format
    {category}.jsonl
    expected_markdown.json
    pdfs/{group}/
    _groundtruth/{group}/*.md    # the reference transcriptions — review these
  output/<pipeline>/         # inference + evaluation results
  reports/                   # comparison_report.{html,md,json}
```

`init` writes a `.env` listing exactly the credentials the selected pipelines
read — discovered by scanning provider source, so it stays correct as providers
are added. `parse-bench customer keys ./acme` shows which are still missing.

## Where ground truth comes from

This is the question that decides whether the evaluation is credible, so it is
worth being precise about what the tool does and does not do.

**The model transcribes. ParseBench derives the rules.**

A vision model (Gemini 3 Pro by default, via OpenRouter) is given one job per
page: transcribe it into markdown, preserving table structure as HTML and
marking formatting that carries meaning. It is never asked to author rules, and
it is never asked to judge a parser's output.

From that transcription, deterministic code derives the evaluation rules using
*the evaluator's own tokenizers*:

| Dimension | Derived from the reference | Scored by |
|---|---|---|
| Content faithfulness | word / sentence / digit bags, reading-order pairs | `content_faithfulness` |
| Semantic formatting | bold, strikethrough, super/subscript spans, headings | `semantic_formatting` |
| Tables | the reference markdown itself | GriTS + TableRecordMatch |
| Charts | data points read from each chart | `ChartDataPointMatch` |

Two consequences worth stating on a call:

1. **Scoring stays deterministic.** ParseBench does not use LLM-as-a-judge, and
   this workflow does not change that. A model wrote the reference document; no
   model scores anything.
2. **Bags are built with the same code that reads them.** Expected and actual
   text are normalized identically, so a rule can never fail because the two
   sides disagreed about what counts as a word.

Everything generated is written with `verified: false` and tagged `bootstrap`,
and the report states the human-verified percentage in its header. Nobody can
mistake a bootstrapped label for a confirmed one.

### Visual grounding is not bootstrapped

Bounding-box ground truth from a VLM is not accurate enough to defend, so the
`layout` dimension is deliberately excluded. If a customer cares about visual
grounding, that needs hand-drawn boxes in `apps/annotator` or their own
coordinates. Say this plainly rather than shipping numbers you would lose an
argument about.

### Cost

`--dry_run` prints the estimate before anything is spent:

```bash
uv run parse-bench customer groundtruth ./acme --dry_run
```

Budget roughly **$0.70 per page** — one transcription call, plus a second call
per page when charts are in scope. For a 40-document evaluation at 5 pages each,
that is around $140.

This number sounds alarming until it is framed correctly: it is the cost of
*manufacturing reference data*, not of parsing. Production parsing is two to
three orders of magnitude cheaper. Frontier-model transcription is a fine way to
build an eval set and a terrible way to run a pipeline — which is itself a useful
thing for a prospect to hear.

Documents longer than `max_pages_per_doc` (default 20) are truncated to their
first N pages *before* parsing, so the evaluated document and the ground truth
cover the same pages. Without that, a parser would be penalised for faithfully
transcribing pages the reference never described.

### Reviewing before you run

The references land in `data/_groundtruth/<group>/*.md` as plain markdown. Read
them. On a customer call, opening one and reading it next to the source PDF is
the single most persuasive thing you can do — it makes the ground truth
inspectable rather than asserted.

Correct anything wrong and re-run `customer run`; regenerating is not needed,
since rules are derived from the reference files on disk.

For structured review, `apps/annotator` opens the same dataset and can mark
rules verified.

## Bringing your own ground truth

If the customer already has labels, skip `groundtruth` entirely and write the
dataset yourself. The format is the public ParseBench format — one JSON object
per line in `data/{category}.jsonl`:

```json
{"pdf": "pdfs/text/policy_01.pdf", "category": "text_content", "id": "policy_01::order::1",
 "type": "order", "verified": true, "tags": ["customer"],
 "rule": {"before": "the insured party is acme limited", "after": "coverage begins on 1 january 2026"}}
```

Plus `data/expected_markdown.json` mapping each `pdf` path to its reference
markdown. `apps/annotator/rule_definitions.json` documents every rule type and
its fields. Anything that loads through `parse_bench.test_cases.loader` works.

A hybrid is often the best answer: bootstrap the bulk, have the customer
hand-verify a held-out subset, and report both. If the two agree, the
bootstrapped ground truth has earned its credibility on their data specifically.

## Reading the report

`reports/comparison_report.html` is the artifact to walk through; the markdown
version is for pasting into email.

Every pipeline is scored on the same documents, so comparisons are **paired** —
each document contributes one score per pipeline, and the difference is taken
per document. That controls for the fact that some documents are simply harder,
which is exactly the confound that makes eyeballed comparisons useless.

Per challenger, against the baseline:

- **Δ vs baseline with a 95% CI** — percentile bootstrap, 10,000 resamples, fixed
  seed. Re-running the report gives identical numbers.
- **p-value** — two-sided Wilcoxon signed-rank, Holm-corrected across the
  challengers in that dimension. Non-parametric because parse scores are
  bounded, skewed, and tie-heavy, which breaks the t-test's assumptions.
- **win / loss / tie counts** — the honest picture when a mean hides a bimodal
  split: excellent on half the documents, terrible on the other half.
- **effect size** — Cohen's *dz*.

Fewer than 10 documents in a dimension is flagged **underpowered**. The numbers
are still shown, but they should not settle an argument. If a customer brings
five documents, the right move is to ask for more, not to report a win.

### What to claim, and what not to

The report's own verdict lines are written to be read aloud without
over-claiming. Use them.

- "No significant difference" means exactly that. Saying "we were slightly
  ahead" from a non-significant 0.4-point gap is how technical credibility gets
  spent for nothing.
- A significant win in one dimension is a win in that dimension. The overall
  number is a macro-average across dimensions, weighting each equally, so it does
  not automatically follow.
- If the customer's data is genuinely a weak spot, the report will say so. That
  is worth more than a rigged demo — it tells the product team something true,
  and prospects can tell the difference.

## Objections you will hear

**"Your model wrote your ground truth."** Correct, and it is written on the
report. The alternative on offer is no ground truth at all, or a competitor's
output treated as truth — which assumes the answer. Then invite the check that
settles it: hand-verify a sample and compare. The references are plain markdown
sitting on their disk; the argument is auditable rather than rhetorical.

**"Gemini favours you."** The reference is a transcription of their document, not
a parse in anyone's format. Offer to regenerate with a different model
(`--model`) and re-run — the rules are derived from the reference, so nothing
else changes. If the ranking holds across two independent reference models, that
is a strong result. If it doesn't, we need to know that before a customer finds
out.

**"These metrics are yours."** They are, and they are public, deterministic, and
readable: the repo is open source, the rules are in the JSONL, and every score
can be recomputed from the output directory. Contrast this with a comparison
nobody can reproduce.

**"Can we use our own held-out set?"** Yes, and encourage it. Bootstrap the bulk,
have them hand-label a held-out subset, and report both.

## Adding a competitor pipeline

If a prospect names a tool that isn't in the registry, it can be added with
Claude Code:

```bash
/integrate-pipeline <name> <API docs or SDK link>
```

See [`.claude/commands/integrate-pipeline.md`](../.claude/commands/integrate-pipeline.md).
Then add the pipeline name to `parsebench_customer.json` and re-run.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No documents found in docs/` | Files are an unsupported type, or nested under an unrecognised subdirectory. Supported: PDF, PNG, JPG, JFIF, DOCX. |
| One pipeline fails, others succeed | Usually a missing key or SDK. `customer run` deliberately continues past a failed pipeline; check `output/<pipeline>/_errors.json`. |
| Semantic formatting scores 0.0 for everything | Expected when the parsers under test emit plain text. It is a real finding, not a bug — plain-text extractors lose formatting that carries meaning. |
| A dimension is missing from the report | No pipeline produced scores for it. Check that documents were sorted into the right `docs/` subdirectory. |
| `OPENROUTER_API_KEY is not set` | Only needed for `groundtruth`. Set it in the project's `.env`, pass `--api_key`, or supply your own ground truth. |
