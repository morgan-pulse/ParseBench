"""Project scaffolding, config, and API-key discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from parse_bench.customer.project import (
    CONFIG_FILENAME,
    CustomerProjectConfig,
    ProjectError,
    ProjectPaths,
    _is_optional_env_var,
    env_status,
    load_project,
    new_config,
    render_env_template,
    required_env_vars,
    save_project,
)


class TestConfig:
    def test_baseline_defaults_to_first_pipeline(self) -> None:
        config = new_config("Acme", ["a", "b", "c"])
        assert config.resolved_baseline() == "a"
        assert config.challengers() == ["b", "c"]

    def test_explicit_baseline_is_excluded_from_challengers(self) -> None:
        config = new_config("Acme", ["a", "b", "c"], baseline="b")
        assert config.resolved_baseline() == "b"
        assert config.challengers() == ["a", "c"]

    def test_empty_pipelines_has_no_baseline(self) -> None:
        config = CustomerProjectConfig(name="Acme", pipelines=[])
        assert config.resolved_baseline() is None
        assert config.challengers() == []

    def test_roundtrip_through_disk(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        original = new_config("Acme", ["llamaparse_agentic"], categories=["table"])
        original.notes = "EU tenant"
        save_project(original, paths)

        loaded, loaded_paths = load_project(tmp_path)
        assert loaded.name == "Acme"
        assert loaded.categories == ["table"]
        assert loaded.notes == "EU tenant"
        assert loaded_paths.config_file == tmp_path / CONFIG_FILENAME

    def test_missing_project_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(ProjectError, match="customer init"):
            load_project(tmp_path)

    def test_corrupt_config_is_reported_not_swallowed(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(ProjectError, match="Invalid project config"):
            load_project(tmp_path)


class TestPaths:
    def test_ensure_dirs_creates_the_skeleton(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path / "acme")
        paths.ensure_dirs()
        for directory in (paths.docs_dir, paths.data_dir, paths.pdfs_dir, paths.output_dir, paths.reports_dir):
            assert directory.is_dir()
        assert paths.group_docs_dir("table").is_dir()

    def test_category_jsonl_matches_dataset_layout(self, tmp_path: Path) -> None:
        paths = ProjectPaths(tmp_path)
        assert paths.category_jsonl("text_content") == tmp_path / "data" / "text_content.jsonl"


class TestEnvDiscovery:
    def test_llamaparse_key_is_found(self) -> None:
        # Discovery reads provider source, so it must work whether or not the
        # vendor SDK is installed in this environment.
        assert "LLAMA_CLOUD_API_KEY" in required_env_vars("llamaparse_agentic")

    def test_unknown_pipeline_returns_empty(self) -> None:
        assert required_env_vars("no_such_pipeline_xyz") == []

    def test_base_url_overrides_are_optional(self) -> None:
        assert _is_optional_env_var("LLAMA_CLOUD_BASE_URL")
        assert _is_optional_env_var("LLAMA_CLOUD_STAGING_API_KEY")
        assert _is_optional_env_var("LLAMA_CLOUD_EU_API_KEY")
        assert not _is_optional_env_var("LLAMA_CLOUD_API_KEY")

    def test_azure_endpoint_counts_as_a_credential(self) -> None:
        # The Azure endpoint is not an override — without it the pipeline
        # cannot run at all, so the customer must be told to supply it.
        assert not _is_optional_env_var("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

    def test_env_status_flags_missing_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
        status = env_status(["llamaparse_agentic"])
        assert "LLAMA_CLOUD_API_KEY" in status["llamaparse_agentic"]["missing"]

    def test_env_status_clears_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "llx-test")
        status = env_status(["llamaparse_agentic"])
        assert status["llamaparse_agentic"]["missing"] == []

    def test_ground_truth_key_is_reported_separately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        config = new_config("Acme", ["llamaparse_agentic"])
        status = env_status(config.pipelines, config.groundtruth)
        assert status["(ground truth)"]["missing"] == ["OPENROUTER_API_KEY"]


class TestEnvTemplate:
    def test_template_lists_only_selected_pipelines(self) -> None:
        config = new_config("Acme", ["llamaparse_agentic"])
        template = render_env_template(["llamaparse_agentic"], config.groundtruth)
        assert "LLAMA_CLOUD_API_KEY=" in template
        assert "OPENROUTER_API_KEY=" in template
        # A key for a pipeline that was not selected must not appear.
        assert "REDUCTO_API_KEY" not in template

    def test_optional_keys_are_marked(self) -> None:
        config = new_config("Acme", ["llamaparse_agentic"])
        template = render_env_template(["llamaparse_agentic"], config.groundtruth)
        base_url_line = next(line for line in template.splitlines() if line.startswith("LLAMA_CLOUD_BASE_URL"))
        assert "optional" in base_url_line


class TestEnvTemplateDeduplication:
    def test_shared_vendor_key_is_emitted_once(self) -> None:
        # Two LlamaParse pipelines read the same key. Emitting it twice reads
        # as "set this twice" and invites two different values under one name,
        # where the last assignment silently wins.
        config = new_config("Acme", ["llamaparse_agentic", "llamaparse_cost_effective"])
        template = render_env_template(["llamaparse_agentic", "llamaparse_cost_effective"], config.groundtruth)
        assignments = [line for line in template.splitlines() if line.startswith("LLAMA_CLOUD_API_KEY=")]
        assert len(assignments) == 1

    def test_the_sharing_pipeline_is_still_listed(self) -> None:
        config = new_config("Acme", ["llamaparse_agentic", "llamaparse_cost_effective"])
        template = render_env_template(["llamaparse_agentic", "llamaparse_cost_effective"], config.groundtruth)
        # The customer must still see that the second pipeline is covered.
        assert "llamaparse_cost_effective" in template

    def test_distinct_vendors_each_get_their_key(self) -> None:
        config = new_config("Acme", ["reducto", "extend_parse"])
        template = render_env_template(["reducto", "extend_parse"], config.groundtruth)
        assert "REDUCTO_API_KEY=" in template
        assert "EXTEND_API_KEY=" in template
