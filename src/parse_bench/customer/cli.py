"""Command-line interface for the customer evaluation workflow."""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path
from typing import Any

from parse_bench.customer.comparison.report import build_report_data, write_reports
from parse_bench.customer.comparison.scores import load_scores
from parse_bench.customer.groundtruth.emit import dataset_summary
from parse_bench.customer.groundtruth.generate import estimate_cost, generate_ground_truth
from parse_bench.customer.ingest import ingest as ingest_documents
from parse_bench.customer.ingest import read_manifest, staged_documents
from parse_bench.customer.project import (
    BOOTSTRAPPABLE_CATEGORIES,
    DEFAULT_PIPELINES,
    DOC_GROUPS,
    MANUAL_ONLY_CATEGORIES,
    ProjectError,
    ProjectPaths,
    env_status,
    load_project,
    new_config,
    render_env_template,
    save_project,
)

_RULE = "=" * 60


def _load(path: str | Path) -> tuple[Any, ProjectPaths]:
    """Load a project and its .env.

    The top-level CLI only looks for a .env in the working directory and the
    repo root, but a customer project keeps its keys next to its documents —
    which is the whole point of a self-contained project directory.
    """
    config, paths = load_project(path)
    if paths.env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(paths.env_file, override=False)
    return config, paths


def has_ground_truth(paths: ProjectPaths) -> bool:
    """Whether the project has ground truth the loader can read.

    Both dataset formats count. Customers bringing their own labels often
    already have them as sidecar ``.test.json`` files next to the PDFs, and
    the stock loader reads that layout — requiring JSONL would reject valid
    ground truth for no reason.
    """
    if not paths.data_dir.exists():
        return False
    if any(paths.data_dir.glob("*.jsonl")):
        return True
    return any(paths.data_dir.rglob("*.test.json"))


def _as_list(value: str | tuple[str, ...] | list[str] | None) -> list[str]:
    """Normalize Fire's flexible list arguments into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _print_header(title: str) -> None:
    print("\n" + _RULE)
    print(title)
    print(_RULE + "\n")


def _customer_readme(name: str, pipelines: list[str]) -> str:
    pipeline_list = "\n".join(f"  - {p}" for p in pipelines)
    return f"""\
# Parsing evaluation — {name}

This directory is a self-contained document-parsing evaluation. Everything runs
on this machine. Your documents are never uploaded anywhere except the parsing
APIs you configure below, which you control.

## 1. Add your documents

Drop files into `docs/`. Sorting them helps target the evaluation, but is optional:

    docs/table/   documents whose tables matter most
    docs/chart/   documents with charts and graphs
    docs/text/    everything else (this is the default)

PDFs, PNGs, JPEGs, and DOCX files are supported. 30–50 documents per dimension is
plenty; the statistics get meaningful somewhere around 10.

## 2. Add API keys

Edit `.env`. It already lists exactly the keys needed for the pipelines selected
for this evaluation:

{pipeline_list}

## 3. Ground truth

If you already have ground truth, see `docs/customer_eval.md` in the ParseBench
repository for the format. Otherwise ParseBench can bootstrap it:

    parse-bench customer groundtruth . --dry_run   # what it will cost
    parse-bench customer groundtruth .

This transcribes each page with a frontier vision model and derives the
evaluation rules from that transcription. The transcriptions land in
`data/_groundtruth/` as plain markdown — read them. Correct anything wrong before
running the evaluation; the whole comparison rests on them.

## 4. Run and report

    parse-bench customer run .
    parse-bench customer report .

The report lands in `reports/comparison_report.html`.
"""


class CustomerCLI:
    """Guided evaluation on a customer's own documents.

    Commands, in order:
        init         Scaffold a project directory
        ingest       Stage documents from docs/ into the dataset
        groundtruth  Bootstrap ground truth from the staged documents
        run          Evaluate every configured pipeline
        report       Statistical comparison report
        status       What is done, what is missing
        keys         Which API keys are needed and which are set
    """

    def init(
        self,
        path: str | Path,
        name: str | None = None,
        pipelines: str | tuple[str, ...] | list[str] | None = None,
        categories: str | tuple[str, ...] | list[str] | None = None,
        baseline: str | None = None,
        force: bool = False,
    ) -> int:
        """Scaffold a customer evaluation project.

        Args:
            path: Directory to create the project in.
            name: Customer name for report titles (default: directory name).
            pipelines: Pipelines to evaluate, comma-separated.
            categories: Dimensions to evaluate, comma-separated.
            baseline: Pipeline everything is compared against (default: first).
            force: Overwrite an existing project config.

        Example:
            parse-bench customer init ./acme --name Acme --pipelines llamaparse_agentic,extend_parse
        """
        from parse_bench.inference.pipelines import list_pipelines

        root = Path(path)
        paths = ProjectPaths(root)
        if paths.config_file.exists() and not force:
            print(f"Project already exists at {paths.config_file}. Use --force to overwrite.", file=sys.stderr)
            return 1

        selected = _as_list(pipelines) or list(DEFAULT_PIPELINES)
        available = set(list_pipelines())
        unknown = [p for p in selected if p not in available]
        if unknown:
            print(f"Unknown pipeline(s): {', '.join(unknown)}", file=sys.stderr)
            print("Run `parse-bench pipelines` to see the full list.", file=sys.stderr)
            return 1

        selected_categories = _as_list(categories) or list(BOOTSTRAPPABLE_CATEGORIES)
        manual = [c for c in selected_categories if c in MANUAL_ONLY_CATEGORIES]
        if manual:
            print(
                f"Note: {', '.join(manual)} cannot be bootstrapped from a model and needs "
                f"hand-drawn ground truth (apps/annotator). Keeping it in the config; "
                f"`groundtruth` will skip it."
            )

        if baseline and baseline not in selected:
            print(f"Baseline '{baseline}' is not in the pipeline list.", file=sys.stderr)
            return 1

        config = new_config(
            name=name or root.resolve().name,
            pipelines=selected,
            categories=selected_categories,
            baseline=baseline,
        )
        paths.ensure_dirs()
        save_project(config, paths)

        if not paths.env_file.exists() or force:
            paths.env_file.write_text(
                render_env_template(selected, config.groundtruth),
                encoding="utf-8",
            )
        (paths.root / "README.md").write_text(_customer_readme(config.name, selected), encoding="utf-8")

        _print_header(f"Project ready: {paths.root}")
        print(f"Customer:  {config.name}")
        print(f"Pipelines: {', '.join(selected)}")
        print(f"Baseline:  {config.resolved_baseline()}")
        print(f"Dimensions: {', '.join(selected_categories)}")
        print("\nNext steps:")
        print(f"  1. Put documents in {paths.docs_dir}/ ({'/'.join(DOC_GROUPS)} subdirectories are optional)")
        print(f"  2. Fill in API keys in {paths.env_file}")
        print(f"  3. parse-bench customer ingest {paths.root}")
        return 0

    def ingest(
        self,
        path: str | Path,
        group: str | None = None,
        force: bool = False,
    ) -> int:
        """Stage documents from docs/ into the evaluation dataset.

        Args:
            path: Project directory.
            group: Force every document into this group (table, chart, text).
            force: Re-copy documents that are already staged.
        """
        try:
            config, paths = _load(path)
        except ProjectError as e:
            print(str(e), file=sys.stderr)
            return 1

        if group and group not in DOC_GROUPS:
            print(f"Unknown group '{group}'. Expected one of: {', '.join(DOC_GROUPS)}", file=sys.stderr)
            return 1

        _print_header("Ingesting documents")
        result = ingest_documents(
            paths,
            group_override=group,
            force=force,
            max_pages=config.groundtruth.max_pages_per_doc,
        )

        if not result.docs:
            print(f"No documents found in {paths.docs_dir}.")
            print(f"Drop PDFs or images there, optionally under {'/'.join(DOC_GROUPS)} subdirectories.")
            return 1

        for group_name, docs in sorted(result.by_group().items()):
            pages = sum(d.pages or 1 for d in docs)
            print(f"  {group_name:<8} {len(docs):>4} document(s), {pages:>5} page(s)")

        for source, original, kept in result.truncated:
            print(f"  truncated {source.name}: {original} pages -> first {kept} (max_pages_per_doc)")
        for source, reason in result.skipped:
            print(f"  skipped {source.name}: {reason}")

        print(f"\nTotal: {len(result.docs)} document(s), {result.total_pages} page(s)")
        print(f"\nNext: parse-bench customer groundtruth {paths.root} --dry_run")
        return 0

    def groundtruth(
        self,
        path: str | Path,
        model: str | None = None,
        dry_run: bool = False,
        force: bool = False,
        max_concurrent: int = 4,
        api_key: str | None = None,
    ) -> int:
        """Bootstrap ground truth for the staged documents.

        A vision model transcribes each page; ParseBench derives the evaluation
        rules from that transcription with the same tokenizers it scores with.
        No model is involved in scoring.

        Args:
            path: Project directory.
            model: Override the ground-truth model for this run.
            dry_run: Print the cost estimate and stop.
            force: Regenerate references that already exist.
            max_concurrent: Documents transcribed in parallel.
            api_key: Ground-truth model key, if not taken from the environment.
        """
        try:
            config, paths = _load(path)
        except ProjectError as e:
            print(str(e), file=sys.stderr)
            return 1

        if model:
            config.groundtruth.model = model
            save_project(config, paths)

        if not staged_documents(paths):
            print(f"No staged documents. Run: parse-bench customer ingest {paths.root}", file=sys.stderr)
            return 1

        estimate = estimate_cost(paths, config)
        _print_header("Ground-truth generation")
        print(f"Model:      {config.groundtruth.model}")
        print(f"Documents:  {estimate.documents}")
        print(f"Pages:      {estimate.pages}")
        print(f"Model calls: {estimate.model_calls} (transcription + chart reading)")
        print(f"Estimated:  ${estimate.estimated_usd:,.2f} at ${estimate.cost_per_page_usd:.2f}/call")
        print("\nThis is an evaluation cost, not a production one. Generating reference")
        print("ground truth is deliberately expensive; parsing at scale is not.")

        if dry_run:
            print("\n(dry run — nothing was generated)")
            return 0

        client = None
        if api_key:
            from parse_bench.customer.groundtruth.client import VisionModelClient

            gt = config.groundtruth
            client = VisionModelClient(model=gt.model, base_url=gt.base_url, api_key_env=gt.api_key_env)
            client.set_api_key(api_key)

        print()
        try:
            result = generate_ground_truth(
                paths,
                config,
                client=client,
                force=force,
                max_concurrent=max_concurrent,
            )
        except Exception as e:
            print(f"\nGround-truth generation failed: {e}", file=sys.stderr)
            return 1

        _print_header("Ground truth generated")
        for category, count in sorted(result.rule_counts.items()):
            print(f"  {category:<18} {count:>7} rule row(s)")
        print(f"\nDocuments: {len(result.documents)}")
        print(f"Tokens:    {result.prompt_tokens:,} in / {result.completion_tokens:,} out")

        if result.dropped_rules:
            print(f"\nDropped {len(result.dropped_rules)} invalid rule(s):")
            for rule_id, reason in result.dropped_rules[:5]:
                print(f"  {rule_id}: {reason}")
        if result.failures:
            print(f"\n{len(result.failures)} document(s) failed:")
            for doc, reason in result.failures:
                print(f"  {Path(doc).name}: {reason}")

        print(f"\nReferences written to {paths.data_dir / '_groundtruth'}")
        print("Review them before running the evaluation — every score depends on them.")
        print(f"\nNext: parse-bench customer run {paths.root}")
        return 0

    def run(
        self,
        path: str | Path,
        pipelines: str | tuple[str, ...] | list[str] | None = None,
        max_concurrent: int = 20,
        force: bool = False,
        skip_inference: bool = False,
        verbose: bool = False,
    ) -> int:
        """Evaluate every configured pipeline on the customer's dataset.

        One pipeline failing does not stop the others: a missing API key for one
        vendor should not cost the whole run.

        Args:
            path: Project directory.
            pipelines: Override the configured pipeline list.
            max_concurrent: Concurrent inference requests per pipeline.
            force: Re-run inference even when results exist.
            skip_inference: Only re-evaluate existing inference results.
            verbose: Verbose pipeline output.
        """
        try:
            config, paths = _load(path)
        except ProjectError as e:
            print(str(e), file=sys.stderr)
            return 1

        selected = _as_list(pipelines) or config.pipelines
        if not selected:
            print("No pipelines configured.", file=sys.stderr)
            return 1

        if not has_ground_truth(paths):
            print(
                f"No ground truth found in {paths.data_dir}. Run: parse-bench customer groundtruth {paths.root}, "
                f"or supply your own (JSONL or sidecar .test.json — see docs/customer_eval.md).",
                file=sys.stderr,
            )
            return 1

        from parse_bench.pipeline.cli import PipelineCLI

        pipeline_cli = PipelineCLI()
        failures: list[tuple[str, int]] = []

        for index, pipeline in enumerate(selected, start=1):
            _print_header(f"Pipeline {index}/{len(selected)}: {pipeline}")
            try:
                exit_code = pipeline_cli.run(
                    pipeline=pipeline,
                    input_dir=paths.data_dir,
                    output_dir=paths.output_dir,
                    max_concurrent=max_concurrent,
                    force=force,
                    verbose=verbose,
                    open_report=False,
                    skip_inference=skip_inference,
                )
            except Exception as e:
                print(f"{pipeline} failed: {e}", file=sys.stderr)
                failures.append((pipeline, 1))
                continue
            if exit_code != 0:
                failures.append((pipeline, exit_code))

        _print_header("Run complete")
        succeeded = [p for p in selected if p not in {name for name, _ in failures}]
        print(f"Succeeded: {len(succeeded)}/{len(selected)}")
        for pipeline, exit_code in failures:
            print(f"  FAILED {pipeline} (exit {exit_code})")

        if not succeeded:
            return 1
        print(f"\nNext: parse-bench customer report {paths.root}")
        return 0

    def report(
        self,
        path: str | Path,
        open_report: bool = True,
    ) -> int:
        """Generate the statistical comparison report.

        Args:
            path: Project directory.
            open_report: Open the HTML report in a browser.
        """
        try:
            config, paths = _load(path)
        except ProjectError as e:
            print(str(e), file=sys.stderr)
            return 1

        scores = load_scores(paths, config.pipelines)
        if not scores:
            print(
                f"No evaluation results found in {paths.output_dir}. Run: parse-bench customer run {paths.root}",
                file=sys.stderr,
            )
            return 1

        data = build_report_data(config, scores, ground_truth=_ground_truth_provenance(paths, config))
        written = write_reports(paths, data)

        _print_header(f"Report: {config.name}")
        for category in data["categories"]:
            print(f"\n{category['label']} ({category['metric']})")
            for summary in category["summaries"]:
                marker = " *" if summary["pipeline"] == data["baseline"] else "  "
                print(f" {marker} {summary['pipeline']:<40} {summary['mean'] * 100:6.1f}  (n={summary['n']})")
            for row in category["comparisons"]:
                # Name the challenger: with several of them, an unattributed
                # verdict line is ambiguous about who it is describing.
                print(f"      {row['challenger']} vs baseline — {category['verdicts'][row['challenger']]}")

        print(f"\nHTML:     {written['html']}")
        print(f"Markdown: {written['markdown']}")
        print(f"JSON:     {written['json']}")

        if open_report:
            webbrowser.open(f"file://{written['html'].absolute()}")
        return 0

    def status(self, path: str | Path) -> int:
        """Show what is done and what is still missing.

        Args:
            path: Project directory.
        """
        try:
            config, paths = _load(path)
        except ProjectError as e:
            print(str(e), file=sys.stderr)
            return 1

        _print_header(f"Project: {config.name}")
        print(f"Directory:  {paths.root}")
        print(f"Pipelines:  {', '.join(config.pipelines)}")
        print(f"Baseline:   {config.resolved_baseline()}")
        print(f"Dimensions: {', '.join(config.categories)}")

        docs = staged_documents(paths)
        manifest = read_manifest(paths) or {}
        print("\nDocuments")
        if docs:
            for group in sorted({d.group for d in docs}):
                group_docs = [d for d in docs if d.group == group]
                pages = sum(d.pages or 1 for d in group_docs)
                print(f"  {group:<8} {len(group_docs):>4} staged, {pages:>5} page(s)")
        else:
            # A bring-your-own-ground-truth project keeps its documents in
            # data/ alongside their labels and never runs ingest. Reporting
            # "none" there would look like the project is empty.
            supplied = sorted(paths.data_dir.rglob("*.pdf")) if paths.data_dir.exists() else []
            if supplied:
                print(f"  {len(supplied):>4} supplied with your own ground truth (not staged via ingest)")
            else:
                print(f"  none staged — put documents in {paths.docs_dir} and run `customer ingest`")
        truncated = manifest.get("truncated") or []
        if truncated:
            print(f"  {len(truncated)} document(s) truncated to fit max_pages_per_doc")

        print("\nGround truth")
        summary = dataset_summary(paths)
        if summary:
            for category, counts in sorted(summary.items()):
                verified_pct = (counts["verified"] / counts["rules"] * 100) if counts["rules"] else 0.0
                print(
                    f"  {category:<18} {counts['rules']:>7} rule(s), "
                    f"{counts['documents']:>4} doc(s), {verified_pct:5.1f}% verified"
                )
        else:
            print("  none — run `customer groundtruth`")

        print("\nEvaluation results")
        scored = load_scores(paths, config.pipelines)
        any_results = bool(scored)
        for pipeline in config.pipelines:
            # Read the same way the report does, so status can never claim a
            # pipeline has run when the report would find nothing for it.
            categories = sorted(name for name, entry in scored.items() if pipeline in entry.by_pipeline)
            state = ", ".join(categories) if categories else "not run"
            print(f"  {pipeline:<40} {state}")

        print("\nAPI keys")
        self._print_env_status(config)

        print("\nNext step:")
        # Ordered by how far along the project is, furthest first. Ground truth
        # is checked before staged documents because a project that brought its
        # own labels keeps them in data/ and never runs ingest at all.
        if any_results:
            print(f"  parse-bench customer report {paths.root}")
        elif summary:
            print(f"  parse-bench customer run {paths.root}")
        elif not docs:
            print(f"  parse-bench customer ingest {paths.root}")
        else:
            print(f"  parse-bench customer groundtruth {paths.root}")
        return 0

    def keys(self, path: str | Path) -> int:
        """Show which API keys this project needs and which are set.

        Args:
            path: Project directory.
        """
        try:
            config, _ = _load(path)
        except ProjectError as e:
            print(str(e), file=sys.stderr)
            return 1
        _print_header("API keys")
        self._print_env_status(config)
        return 0

    @staticmethod
    def _print_env_status(config: Any) -> None:
        status = env_status(config.pipelines, config.groundtruth)
        for target, info in status.items():
            if not info["vars"]:
                print(f"  {target:<40} no credentials detected")
                continue
            missing = info["missing"]
            mark = "MISSING" if missing else "ok"
            print(f"  {target:<40} {mark:<8} {', '.join(info['vars'])}")


def _ground_truth_provenance(paths: ProjectPaths, config: Any) -> dict[str, Any]:
    """Ground-truth counts and provenance for the report header."""
    summary = dataset_summary(paths)
    total_rules = sum(c["rules"] for c in summary.values())
    verified = sum(c["verified"] for c in summary.values())
    documents = len(staged_documents(paths))
    return {
        "documents": documents,
        "rules": total_rules,
        "verified": verified,
        "verified_pct": (verified / total_rules * 100) if total_rules else 0.0,
        "model": config.groundtruth.model,
        "categories": {name: counts["rules"] for name, counts in sorted(summary.items())},
    }
