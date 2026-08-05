"""Customer-facing evaluation workflow.

Wraps the ParseBench harness in a guided, self-contained flow that a customer
can run entirely inside their own environment, on documents they cannot share:

    parse-bench customer init ./acme
    parse-bench customer ingest ./acme
    parse-bench customer groundtruth ./acme
    parse-bench customer run ./acme
    parse-bench customer report ./acme

Nothing in this package uploads customer documents anywhere except the
parsing APIs the customer explicitly configures, plus the ground-truth model
they choose during `groundtruth` (which can be skipped entirely if they bring
their own labels).
"""

from parse_bench.customer.project import (
    CustomerProjectConfig,
    ProjectPaths,
    load_project,
)

__all__ = [
    "CustomerProjectConfig",
    "ProjectPaths",
    "load_project",
]
