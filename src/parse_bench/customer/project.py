"""Customer project layout, configuration, and API-key discovery.

A "customer project" is a self-contained directory holding one evaluation:
their documents, their ground truth, their pipeline selection, and the run
output. It never leaves their machine.

Layout::

    <project>/
      parsebench_customer.json   # this config
      .env                       # API keys for the selected pipelines
      docs/                      # customer drops source documents here
        table/  chart/  text/    # optional per-dimension subdirectories
      data/                      # generated ParseBench dataset
        pdfs/{table,chart,text}/
        {table,chart,text_content,text_formatting}.jsonl
      output/<pipeline>/         # inference + evaluation results
      reports/                   # statistical comparison report
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

CONFIG_FILENAME = "parsebench_customer.json"

# Evaluation categories this workflow can bootstrap ground truth for.
# `layout` is deliberately excluded: bounding-box ground truth from a VLM is
# not reliable enough to defend in front of a customer. Visual grounding needs
# either the annotator app (apps/annotator) or the customer's own boxes.
BOOTSTRAPPABLE_CATEGORIES: tuple[str, ...] = (
    "table",
    "chart",
    "text_content",
    "text_formatting",
)

# Categories that require human-drawn ground truth rather than model bootstrap.
MANUAL_ONLY_CATEGORIES: tuple[str, ...] = ("layout",)

# Evaluation category -> inference group. Categories sharing a group are parsed
# once and evaluated with different rule sets, matching the public dataset's
# layout (see test_cases/loader.py).
CATEGORY_TO_GROUP: dict[str, str] = {
    "table": "table",
    "chart": "chart",
    "text_content": "text",
    "text_formatting": "text",
    "layout": "layout",
}

# Document source groups a customer can sort documents into under docs/.
DOC_GROUPS: tuple[str, ...] = ("table", "chart", "text")

# Group -> evaluation categories generated from documents in that group.
GROUP_TO_CATEGORIES: dict[str, tuple[str, ...]] = {
    "table": ("table",),
    "chart": ("chart",),
    "text": ("text_content", "text_formatting"),
}

# Documents dropped directly in docs/ with no subdirectory land here. Text is
# the safe default: every document has text, not every document has tables.
DEFAULT_DOC_GROUP = "text"

# Pipelines proposed by `init` when the SA doesn't name any. Chosen to be the
# comparison a prospect actually cares about: us, the incumbent they mentioned,
# and a frontier VLM as a neutral reference point.
DEFAULT_PIPELINES: tuple[str, ...] = (
    "llamaparse_agentic",
    "llamaparse_cost_effective",
)


class GroundTruthConfig(BaseModel):
    """How bootstrap ground truth is generated for this project."""

    model: str = Field(
        default="google/gemini-3-pro",
        description="Model slug used to generate ground truth (OpenRouter naming).",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible base URL for the ground-truth model.",
    )
    api_key_env: str = Field(
        default="OPENROUTER_API_KEY",
        description="Environment variable holding the ground-truth model's API key.",
    )
    dpi: int = Field(default=150, description="Render DPI for page images sent to the model.")
    max_pages_per_doc: int = Field(
        default=20,
        description="Cap on pages bootstrapped per document, to keep cost predictable.",
    )
    estimated_cost_per_page_usd: float = Field(
        default=0.70,
        description="Cost estimate per page, used by --dry_run. Eval-only cost, not a production number.",
    )
    max_rules_per_page: int = Field(
        default=25,
        description="Cap on rules generated per page per category.",
    )


class CustomerProjectConfig(BaseModel):
    """Configuration for one customer evaluation."""

    name: str = Field(description="Customer or engagement name, used in report titles.")
    created_at: str = Field(default="", description="ISO timestamp of project creation.")
    pipelines: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PIPELINES),
        description="Pipelines to evaluate, in report order.",
    )
    baseline_pipeline: str | None = Field(
        default=None,
        description="Pipeline every other one is compared against. Defaults to the first pipeline.",
    )
    categories: list[str] = Field(
        default_factory=lambda: list(BOOTSTRAPPABLE_CATEGORIES),
        description="Evaluation categories to generate ground truth for.",
    )
    groundtruth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)
    notes: str = Field(default="", description="Free-text notes carried into the report.")

    def resolved_baseline(self) -> str | None:
        """Baseline pipeline, falling back to the first configured pipeline."""
        if self.baseline_pipeline:
            return self.baseline_pipeline
        return self.pipelines[0] if self.pipelines else None

    def challengers(self) -> list[str]:
        """Pipelines compared against the baseline."""
        baseline = self.resolved_baseline()
        return [p for p in self.pipelines if p != baseline]


class ProjectPaths:
    """Filesystem layout for a customer project."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def config_file(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def pdfs_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def group_docs_dir(self, group: str) -> Path:
        return self.docs_dir / group

    def group_pdfs_dir(self, group: str) -> Path:
        return self.pdfs_dir / group

    def category_jsonl(self, category: str) -> Path:
        return self.data_dir / f"{category}.jsonl"

    def pipeline_output_dir(self, pipeline: str) -> Path:
        return self.output_dir / pipeline

    def ensure_dirs(self) -> None:
        """Create the directory skeleton."""
        for d in (self.docs_dir, self.data_dir, self.pdfs_dir, self.output_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)
        for group in DOC_GROUPS:
            self.group_docs_dir(group).mkdir(parents=True, exist_ok=True)


class ProjectError(Exception):
    """Raised when a customer project is missing or malformed."""


def load_project(root: str | Path) -> tuple[CustomerProjectConfig, ProjectPaths]:
    """Load a customer project from disk.

    :param root: Project directory.
    :raises ProjectError: If the directory is not an initialized project.
    """
    paths = ProjectPaths(Path(root))
    if not paths.config_file.exists():
        raise ProjectError(
            f"No customer project at {paths.root} (missing {CONFIG_FILENAME}). "
            f"Run: parse-bench customer init {paths.root}"
        )
    try:
        config = CustomerProjectConfig.model_validate_json(paths.config_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise ProjectError(f"Invalid project config at {paths.config_file}: {e}") from e
    return config, paths


def save_project(config: CustomerProjectConfig, paths: ProjectPaths) -> None:
    """Write the project config back to disk."""
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.config_file.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")


def new_config(
    name: str,
    pipelines: list[str],
    categories: list[str] | None = None,
    baseline: str | None = None,
) -> CustomerProjectConfig:
    """Build a fresh project config with a creation timestamp."""
    return CustomerProjectConfig(
        name=name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        pipelines=pipelines,
        baseline_pipeline=baseline,
        categories=categories or list(BOOTSTRAPPABLE_CATEGORIES),
    )


# ── API key discovery ────────────────────────────────────────────────────────

_ENV_PATTERNS = (
    re.compile(r"os\.getenv\(\s*[\"']([A-Z][A-Z0-9_]{2,})[\"']"),
    re.compile(r"os\.environ\.get\(\s*[\"']([A-Z][A-Z0-9_]{2,})[\"']"),
    re.compile(r"os\.environ\[\s*[\"']([A-Z][A-Z0-9_]{2,})[\"']"),
)

# Environment variables that tune behavior rather than grant access. Listing
# these in a customer's .env template would imply they are required.
_NON_CREDENTIAL_ENV_VARS = frozenset(
    {
        "PARSEBENCH_FAST_TEDS",
        "PATH",
        "HOME",
        "PWD",
    }
)


def _is_optional_env_var(name: str) -> bool:
    """Whether a variable is an override rather than a credential to go and fetch.

    Base-URL overrides are optional. So are region and staging key variants:
    a provider reads several of them but a customer only ever needs the one
    matching their deployment, and flagging the rest as MISSING sends them
    hunting for keys that do not exist.
    """
    if "_STAGING_" in name or "_EU_" in name:
        return True
    return name.endswith(("_BASE_URL", "_ENDPOINT")) and "DOCUMENT_INTELLIGENCE" not in name


_REGISTER_PROVIDER_PATTERN = re.compile(r"@register_provider\(\s*[\"']([^\"']+)[\"']")


@lru_cache(maxsize=1)
def _provider_source_files() -> dict[str, Path]:
    """Map provider name -> source file, by scanning source rather than importing.

    Importing is not an option here: provider modules are imported lazily and
    silently skipped when their vendor SDK is absent, so a customer running
    `init` before `uv sync --extra runners` would get an empty key template —
    exactly when they most need to know which keys to go and fetch.
    """
    import parse_bench

    providers_root = Path(parse_bench.__file__).parent / "inference" / "providers"
    mapping: dict[str, Path] = {}
    if not providers_root.exists():
        return mapping
    for source_file in providers_root.rglob("*.py"):
        try:
            source = source_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for provider_name in _REGISTER_PROVIDER_PATTERN.findall(source):
            mapping.setdefault(provider_name, source_file)
    return mapping


def required_env_vars(pipeline_name: str) -> list[str]:
    """Environment variables a pipeline reads, discovered from its provider source.

    Scanning source rather than hardcoding a table means new providers are
    picked up automatically as they land in the repo.

    :param pipeline_name: Registered pipeline name.
    :return: Sorted env var names, or an empty list if the provider can't be resolved.
    """
    from parse_bench.inference.pipelines import get_pipeline

    try:
        spec = get_pipeline(pipeline_name)
    except Exception:
        return []

    source_file = _provider_source_files().get(spec.provider_name)
    if source_file is None:
        return []
    try:
        source = source_file.read_text(encoding="utf-8")
    except Exception:
        return []

    found: set[str] = set()
    for pattern in _ENV_PATTERNS:
        found.update(pattern.findall(source))
    return sorted(v for v in found if v not in _NON_CREDENTIAL_ENV_VARS)


def env_status(pipelines: list[str], groundtruth: GroundTruthConfig | None = None) -> dict[str, dict[str, Any]]:
    """Report which credentials each pipeline needs and whether they are set.

    :return: Mapping of pipeline name -> {"vars": [...], "missing": [...]}.
             The ground-truth model, when given, appears under the key
             ``"(ground truth)"``.
    """
    status: dict[str, dict[str, Any]] = {}
    for pipeline in pipelines:
        variables = required_env_vars(pipeline)
        missing = [v for v in variables if not _is_optional_env_var(v) and not os.getenv(v)]
        status[pipeline] = {"vars": variables, "missing": missing}

    if groundtruth is not None:
        gt_var = groundtruth.api_key_env
        status["(ground truth)"] = {
            "vars": [gt_var],
            "missing": [] if os.getenv(gt_var) else [gt_var],
        }
    return status


def render_env_template(pipelines: list[str], groundtruth: GroundTruthConfig) -> str:
    """Render a .env template listing only the keys this project needs."""
    lines = [
        "# =============================================================================",
        "# ParseBench customer evaluation — API keys",
        "#",
        "# Only the keys below are needed for the pipelines selected in",
        f"# {CONFIG_FILENAME}. Everything runs locally; these keys are used",
        "# solely to call the parsing APIs you chose to evaluate.",
        "# =============================================================================",
        "",
        "# Ground-truth generation (only needed for `customer groundtruth`).",
        "# Skip this if you are supplying your own ground truth.",
        f"{groundtruth.api_key_env}=",
        "",
    ]
    # Pipelines from the same vendor share credentials. Emitting a variable
    # twice reads as "set this twice" and invites a customer to paste two
    # different keys into the same name, where the last one silently wins.
    emitted: set[str] = set()
    for pipeline in pipelines:
        variables = required_env_vars(pipeline)
        new_variables = [v for v in variables if v not in emitted]

        if not variables:
            lines.append(f"# {pipeline}")
            lines.append(f"# (no credentials detected for {pipeline})")
            lines.append("")
            continue

        if not new_variables:
            shared = ", ".join(variables)
            lines.append(f"# {pipeline} — uses {shared} above")
            lines.append("")
            continue

        lines.append(f"# {pipeline}")
        for var in new_variables:
            suffix = "  # optional" if _is_optional_env_var(var) else ""
            lines.append(f"{var}={suffix}")
            emitted.add(var)
        lines.append("")
    return "\n".join(lines)
