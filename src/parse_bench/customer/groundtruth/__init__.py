"""Bootstrap ground truth for customer documents.

The division of labour matters, so it is worth stating plainly:

* The **model** does perception only — it transcribes a page into reference
  markdown (HTML tables, formatting markup preserved) and reads chart data
  points. That is the one thing a VLM is genuinely good at.
* **ParseBench** derives the evaluation rules from that reference, using the
  exact tokenizers the evaluator uses at scoring time
  (``rules_bag.SentenceBagRule`` / ``WordBagRule``). No model is asked to
  author rule JSON, and no model is involved in scoring.

This keeps the benchmark's "no LLM-as-a-judge" property intact: the model
produces a reference document, not a verdict. It also means the bags are
normalized identically on both sides, so a rule can never be unsatisfiable
because of a tokenizer mismatch.

The reference is a bootstrap, not gospel. Everything generated here is written
with ``verified: false`` so it can be reviewed in ``apps/annotator`` and so the
report can state exactly how much of the ground truth a human has confirmed.
"""

from parse_bench.customer.groundtruth.generate import (
    GenerationResult,
    generate_ground_truth,
)

__all__ = ["GenerationResult", "generate_ground_truth"]
