# ParseBench — evaluate document parsers on your own documents

[![Website](https://img.shields.io/badge/Website-parsebench.ai-blue)](https://parsebench.ai)
[![arXiv](https://img.shields.io/badge/arXiv-2604.08538-b31b1b.svg)](https://arxiv.org/abs/2604.08538)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/llamaindex/ParseBench)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

Choosing a document parsing tool usually comes down to a vendor demo on a
handful of pages, or an internal bake-off where "which output looks better?"
gets decided by eye. Neither survives contact with a production workload.

This repository is a benchmark that measures whether parsed output preserves
the structure and meaning an agent needs to act on — and it runs **on your own
documents, on your own machine**. Your files are sent only to the parsing APIs
you choose to evaluate. Nothing else leaves.

```bash
uv sync --extra runners

uv run parse-bench customer init ./my-eval --pipelines llamaparse_agentic,reducto,extend_parse
# drop your documents into ./my-eval/docs/, add API keys to ./my-eval/.env
uv run parse-bench customer ingest ./my-eval
uv run parse-bench customer groundtruth ./my-eval
uv run parse-bench customer run ./my-eval
uv run parse-bench customer report ./my-eval
```

<p align="center">
  <img src="docs/parsebench_teaser.png" alt="ParseBench overview: five capability dimensions" width="100%">
</p>

---

## Why this exists

Three things usually go wrong when a team evaluates parsers on real documents.

**The documents can't be shared.** Contracts, claims files, and medical records
cannot be uploaded to a vendor for a bake-off, so the evaluation either doesn't
happen or happens on unrepresentative sample data. Everything here runs
locally, so the question "can we send you our documents?" never comes up.

**There is no ground truth.** Without it, "which output is better?" is a matter
of taste, and whoever's output is inspected first sets the standard. This
repository can bootstrap ground truth from your documents (below), or use
labels you already have.

**One tool's output gets used as the answer key.** It is a natural shortcut and
it silently decides the result: whichever tool produced the reference wins,
because every difference is scored as the other tool's error. A real reference
has to be independent of every tool being measured.

## What you get

A paired, per-document statistical comparison — not a side-by-side eyeball:

```
Tables (grits_trm_composite)
  * parser_a          80.0  (n=26)      95% CI 71.1–88.4
    parser_b          75.9  (n=26)      95% CI 67.4–84.0
    parser_c          62.6  (n=26)      95% CI 51.2–73.7

  parser_b vs baseline — No significant difference (4.2 point gap, p=0.258, n=26).
  parser_c vs baseline — parser_a is better by 17.4 points
                         (95% CI -27.5 to -8.6, p=0.0009, n=26).
```

Every tool is scored on the same documents, so differences are measured per
document and then aggregated — which controls for the fact that some documents
are simply harder. You get a confidence interval, a significance test, and
win/loss/tie counts, so a 17-point gap and a 4-point gap are not presented as
the same kind of finding. Dimensions with fewer than 10 documents are flagged
**underpowered** rather than reported as wins.

Output lands in `reports/` as HTML (to read), markdown (to paste into an
email), and JSON (to do your own analysis on).

## What gets measured

Five dimensions, each targeting a failure mode that breaks agent workflows:

| Dimension | What it catches | Metric |
|---|---|---|
| **Tables** | Merged cells and hierarchical headers. A misaligned header means the agent reads the wrong column. | GriTS + TableRecordMatch |
| **Content faithfulness** | Omissions, hallucinations, reading-order violations. Incomplete or fabricated context compromises every downstream decision. | Content Faithfulness Score |
| **Semantic formatting** | Formatting that carries meaning — strikethrough, super/subscript, bold, heading hierarchy. A struck-through price is not the current price. | Semantic Formatting Score |
| **Charts** | Exact data points with correct series and axis labels. Most parsers return raw text, leaving agents unable to read precise values. | ChartDataPointMatch |
| **Visual grounding** | Tracing every extracted element back to its location on the page. Required wherever values must be auditable. | Element Pass Rate |

Scoring is deterministic and rule-based throughout. **No language model judges
any output.**

## Where ground truth comes from

If you already have labels, skip this step — `customer groundtruth` is the only
command that calls a model, and without it no document reaches anything except
the parsers you are evaluating. See [bring your own ground truth](#bring-your-own-ground-truth).

Otherwise ParseBench can bootstrap it, and the division of labour is the part
worth understanding:

> **A vision model transcribes each page. ParseBench derives the rules.**

The model is given one job — transcribe this page into markdown, preserving
table structure and meaningful formatting. It is never asked to write
evaluation rules, and it is never asked to judge a parser. From that
transcription, deterministic code derives the rules using *the evaluator's own
tokenizers*, so expected and actual text are normalized identically and a rule
can never fail because the two sides disagreed about what a word is.

Two things follow, and both matter when someone challenges a result:

- **Scoring stays deterministic.** A model wrote a reference *document*; no
  model scores anything.
- **The reference is inspectable.** Transcriptions land in
  `data/_groundtruth/**.md` as plain markdown. Read them. Correct them. The
  next run picks the corrections up — rules are derived from the files on disk.

Everything generated is marked `verified: false` and tagged `bootstrap`, and
the report states the human-verified percentage in its header, so bootstrapped
labels can't be mistaken for confirmed ones.

Budget roughly **$0.70 per page**; `--dry_run` prints the estimate before
anything is spent. That is the cost of manufacturing reference data, not of
parsing — production parsing is orders of magnitude cheaper.

**Visual grounding is not bootstrapped.** Bounding boxes from a vision model
are not accurate enough to rely on, so that dimension needs hand-drawn boxes
via [`apps/annotator`](apps/annotator) or coordinates you already have.

## What leaves your machine

| Step | Network calls |
|---|---|
| `init`, `ingest`, `report` | none |
| `groundtruth` | page images → the model endpoint you configure (skip entirely if you have labels) |
| `run` | your documents → the parsing APIs you selected, and nowhere else |

The ground-truth model is reached over an OpenAI-compatible endpoint, so it can
point at OpenRouter, Azure OpenAI, or an internal gateway — configurable in
`parsebench_customer.json`. Self-hosted parsers can be evaluated with no
outbound calls at all.

## Which tools you can compare

90+ pipelines are registered — hosted APIs, frontier VLMs, open-weight models,
and local libraries. Run `uv run parse-bench pipelines`, or see
[docs/pipelines.md](docs/pipelines.md).

You only need credentials for the tools you actually select. `customer init`
writes a `.env` listing exactly those, discovered from the provider code:

```bash
uv run parse-bench customer keys ./my-eval    # what's needed, what's set
```

If a tool you care about isn't registered, add it with [Claude Code](https://claude.ai/code):

```bash
/integrate-pipeline <name> <API docs or SDK link>
```

## Bring your own ground truth

Both dataset formats load through the stock loader — your numbers stay
comparable to the public benchmark rather than a private fork of it.

**Sidecar** — often what an existing internal benchmark already looks like.
Put `<name>.test.json` next to each PDF under `data/<group>/`:

```json
{"expected_markdown": "...", "tags": ["contracts"],
 "test_rules": [{"type": "present", "text": "...", "max_l_dist": 5}]}
```

**JSONL** — one rule per line in `data/{category}.jsonl`, plus
`data/expected_markdown.json` mapping each PDF to its reference:

```json
{"pdf": "pdfs/text/policy_01.pdf", "category": "text_content",
 "id": "policy_01::order::1", "type": "order", "verified": true,
 "rule": {"before": "the insured party is acme limited",
          "after": "coverage begins on 1 january 2026"}}
```

`apps/annotator/rule_definitions.json` documents every rule type and its
fields. A hybrid is often strongest: bootstrap the bulk, hand-verify a held-out
subset, and report both — agreement between the two earns the bootstrapped
labels their credibility on your data specifically.

> **Reusing an internal benchmark?** If its thresholds were ever loosened while
> looking at one parser's output, that parser has a head start. The check is
> cheap: split the documents into those a recalibration pass touched and those
> it didn't, and confirm the ranking survives on the untouched subset.

## Documentation

| Document | For |
|---|---|
| [`src/parse_bench/customer/README.md`](src/parse_bench/customer/README.md) | How the workflow is built — architecture, why rules are derived rather than authored, the statistics |
| [`docs/customer_eval.md`](docs/customer_eval.md) | Running an evaluation end to end, cost, reading the report, objection handling |
| [`docs/pipelines.md`](docs/pipelines.md) | Every registered parsing tool |

## CLI reference

| Command | Description |
|---|---|
| `parse-bench customer init` | Scaffold an evaluation project |
| `parse-bench customer ingest` | Stage your documents into the dataset |
| `parse-bench customer groundtruth` | Bootstrap ground truth (`--dry_run` to price it) |
| `parse-bench customer run` | Evaluate every configured tool |
| `parse-bench customer report` | Statistical comparison report |
| `parse-bench customer status` | Progress and next step |
| `parse-bench customer keys` | Which credentials are needed and set |
| `parse-bench run` | Run a tool against the public benchmark dataset |
| `parse-bench download` / `status` | Manage the public dataset |
| `parse-bench pipelines` | List available tools |
| `parse-bench compare` / `leaderboard` / `serve` | Compare, rank, and view reports |

Advanced subcommands: `inference`, `evaluation`, `analysis`, `pipeline`, `data`.

<details>
<summary><strong>Project layout</strong></summary>

```
my-eval/
├── parsebench_customer.json     # tools, dimensions, ground-truth settings
├── .env                         # only the keys your selected tools need
├── docs/                        # your documents (table/ chart/ text/ optional)
├── data/                        # generated dataset
│   ├── {category}.jsonl
│   ├── expected_markdown.json
│   ├── pdfs/{group}/
│   └── _groundtruth/**.md       # reference transcriptions — review these
├── output/<tool>/               # inference + evaluation results
└── reports/                     # comparison_report.{html,md,json}
```

</details>

<details>
<summary><strong>Environment variables</strong></summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `PARSEBENCH_FAST_TEDS` | `1` | Fast Zhang-Shasha TEDS table metric (uses `numba` when the `fast` extra is installed, otherwise an exact pure-Python fallback). Set to `0` to force the original APTED implementation — scores are identical either way. |

Add the `fast` extra for a JIT-accelerated table metric:
`uv sync --extra runners --extra fast`

</details>

---

## The public benchmark

The same harness, run on a public dataset — useful as a starting point, and as
a sanity check that a tool behaves on your documents the way it does in the
open.

The dataset covers ~2,000 human-verified pages from real enterprise documents
(insurance, finance, government), hosted on HuggingFace as
[`llamaindex/ParseBench`](https://huggingface.co/datasets/llamaindex/ParseBench):

| Dimension | Metric | Pages | Docs | Rules |
|-----------|--------|------:|-----:|------:|
| Tables | GTRM (GriTS + TableRecordMatch) | 503 | 284 | --- |
| Charts | ChartDataPointMatch | 568 | 99 | 4,864 |
| Content Faithfulness | Content Faithfulness Score | 506 | 506 | 141,322 |
| Semantic Formatting | Semantic Formatting Score | 476 | 476 | 5,997 |
| Visual Grounding | Element Pass Rate | 500 | 321 | 16,325 |
| **Total (unique)** | | **2,078** | **1,211** | **169,011** |

```bash
uv run parse-bench run llamaparse_agentic --test   # 3 files per category
uv run parse-bench run llamaparse_agentic          # full benchmark
uv run parse-bench serve llamaparse_agentic        # view reports
```

<!-- LEADERBOARD:START -->
_Top 10 by Overall score. For the full sortable, filterable leaderboard, see [parsebench.ai](https://parsebench.ai); for raw data, see [leaderboard.csv](leaderboard.csv)._

| Rank | Provider | Category | Overall | Tables | Charts | Content Faith. | Sem. Format. | Visual Ground. | ¢ / Page |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | LlamaParse Agentic | LlamaParse | 84.88 | 90.74 | 78.11 | 89.68 | 85.24 | 80.62 | 1.25¢ |
| 2 | Pulse Ultra 2 | Commercial - Startup APIs | 77.08 | 75.45 | 90.82 | 79.49 | 73.05 | 66.56 | 15.00¢ |
| 3 | LlamaParse Cost Effective | LlamaParse | 76.77 | 81.42 | 70.15 | 90.92 | 68.78 | 72.59 | 0.38¢ |
| 4 | KDL-Frontier-Parser-nano | VLM - Open Weight | 76.36 | 85.56 | 63.41 | 87.19 | 66.81 | 78.84 | — |
| 5 | Extend (2.0) | Commercial - Startup APIs | 75.33 | 84.82 | 78.31 | 84.59 | 60.31 | 68.61 | 2.50¢ |
| 6 | Google Gemini 3 Flash (Thinking High) | VLM - Proprietary | 75.05 | 91.50 | 64.79 | 90.87 | 68.31 | 59.77 | 2.41¢ |
| 7 | Infinity-Parser2-Pro | VLM - Open Weight | 74.28 | 86.4 | 61.3 | 89.7 | 59.1 | 74.9 | — |
| 8 | Extend Light (1.0) | Commercial - Startup APIs | 73.26 | 75.8 | 78.6 | 84.8 | 58.6 | 68.5 | 0.62¢ |
| 9 | Infinity-Parser2-Flash | VLM - Open Weight | 73.25 | 82.88 | 55.56 | 89.52 | 57.7 | 80.61 | — |
| 10 | Reducto (Agentic) | Commercial - Startup APIs | 72.97 | 80.42 | 73.4 | 86.37 | 57.6 | 67.07 | 4.76¢ |
<!-- LEADERBOARD:END -->

A tool is eligible for the public leaderboard if it is publicly accessible
(open weights or a self-serve API), finishes a run in single-digit hours, and
needs no custom framework changes — so the comparison stays fair. Concurrency
is tuned to each provider's recommended settings.

Public scores are a useful prior, not a substitute for measuring on your own
documents: the ranking on enterprise insurance filings is not necessarily the
ranking on your engineering drawings.

## Citation

```bibtex
@misc{zhang2026parsebench,
  title={ParseBench: A Document Parsing Benchmark for AI Agents},
  author={Boyang Zhang and Sebastián G. Acosta and Preston Carlson and Sacha Bron and Pierre-Loïc Doulcet and Daniel B. Ospina and Simon Suo},
  year={2026},
  eprint={2604.08538},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.08538},
}
```

## Links

- **Paper**: [arXiv:2604.08538](https://arxiv.org/abs/2604.08538)
- **HuggingFace Dataset**: [llamaindex/ParseBench](https://huggingface.co/datasets/llamaindex/ParseBench)
- **Code**: [run-llama/ParseBench](https://github.com/run-llama/ParseBench)
