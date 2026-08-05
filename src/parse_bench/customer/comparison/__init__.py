"""Statistical comparison of pipelines on a customer's own documents.

The point of this package is to replace "we eyeballed a few outputs and ours
looked worse" with a paired, per-document comparison that reports an effect
size, a confidence interval, and how many documents actually back it.
"""

from parse_bench.customer.comparison.scores import (
    CategoryScores,
    load_scores,
)
from parse_bench.customer.comparison.stats import (
    ComparisonRow,
    PipelineSummary,
    compare_to_baseline,
)

__all__ = [
    "CategoryScores",
    "ComparisonRow",
    "PipelineSummary",
    "compare_to_baseline",
    "load_scores",
]
