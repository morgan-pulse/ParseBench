"""Fail when a branch adds Ruff or mypy debt relative to a git baseline."""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_MYPY_ERROR_RE = re.compile(r"^(.*?):(\d+): error: (.*?)(?:  \[([^]]+)\])?$")


@dataclass(frozen=True, order=True)
class Diagnostic:
    path: str
    code: str
    message: str
    source: str = field(compare=False)
    row: int = field(compare=False)


def _run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _changed_python_files(root: Path, baseline: str) -> list[str]:
    result = _run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", baseline, "--"],
        cwd=root,
        check=True,
    )
    return sorted(path for path in result.stdout.splitlines() if path.endswith(".py") and (root / path).is_file())


def _extract_baseline(root: Path, baseline: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", baseline],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def _source_line(root: Path, relative_path: str, row: int) -> str:
    try:
        lines = (root / relative_path).read_text().splitlines()
    except (OSError, UnicodeError):
        return ""
    return lines[row - 1].strip() if 0 < row <= len(lines) else ""


def _relative_path(filename: str, root: Path) -> str:
    path = Path(filename)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _normalized_message(message: str) -> str:
    return re.sub(r"\bfrom line \d+\b", "from prior line", message)


def _ruff_diagnostics(root: Path, files: list[str], ruff: str) -> list[Diagnostic]:
    if not files:
        return []
    result = _run([ruff, "check", "--output-format=json", *files], cwd=root)
    payload: list[dict[str, Any]] = json.loads(result.stdout or "[]")
    diagnostics: list[Diagnostic] = []
    for item in payload:
        path = _relative_path(str(item["filename"]), root)
        row = int(item["location"]["row"])
        diagnostics.append(
            Diagnostic(
                path=path,
                code=str(item["code"]),
                message=_normalized_message(str(item["message"])),
                source=_source_line(root, path, row),
                row=row,
            )
        )
    return diagnostics


def _mypy_diagnostics(root: Path, files: list[str], mypy: str) -> list[Diagnostic]:
    if not files:
        return []
    result = _run([mypy, "--follow-imports=skip", *files], cwd=root)
    diagnostics: list[Diagnostic] = []
    for line in result.stdout.splitlines():
        match = _MYPY_ERROR_RE.match(line)
        if match is None:
            continue
        filename, row_text, message, code = match.groups()
        path = _relative_path(filename, root)
        row = int(row_text)
        diagnostics.append(
            Diagnostic(
                path=path,
                code=code or "mypy",
                message=_normalized_message(message),
                source=_source_line(root, path, row),
                row=row,
            )
        )
    return diagnostics


def _added_line_ranges(root: Path, baseline: str, files: list[str]) -> dict[str, list[range]]:
    result = _run(["git", "diff", "--unified=0", baseline, "--", *files], cwd=root, check=True)
    ranges: dict[str, list[range]] = {}
    current_path: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_path = line[6:]
            continue
        match = _HUNK_RE.match(line)
        if match is None or current_path is None:
            continue
        start = int(match.group(3))
        count = int(match.group(4) or "1")
        if count:
            ranges.setdefault(current_path, []).append(range(start, start + count))
    return ranges


def _format_hunks(root: Path, files: list[str], ruff: str) -> list[tuple[str, range]]:
    if not files:
        return []
    result = _run([ruff, "format", "--diff", *files], cwd=root)
    hunks: list[tuple[str, range]] = []
    current_path: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("--- "):
            current_path = _relative_path(line[4:].split("\t", 1)[0], root)
            continue
        match = _HUNK_RE.match(line)
        if match is None or current_path is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        hunks.append((current_path, range(start, start + max(count, 1))))
    return hunks


def _ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


def _new_diagnostics(
    current: Counter[Diagnostic],
    baseline: Counter[Diagnostic],
) -> Counter[Diagnostic]:
    return current - baseline


def _on_changed_lines(
    diagnostics: Iterable[Diagnostic],
    changed_ranges: dict[str, list[range]],
) -> Counter[Diagnostic]:
    return Counter(
        diagnostic
        for diagnostic in diagnostics
        if any(diagnostic.row in changed for changed in changed_ranges.get(diagnostic.path, []))
    )


def _print_diagnostics(label: str, diagnostics: Counter[Diagnostic]) -> None:
    for diagnostic, count in sorted(diagnostics.items()):
        suffix = f" (x{count})" if count > 1 else ""
        print(f"{label}: {diagnostic.path}: {diagnostic.code}: {diagnostic.message} :: {diagnostic.source}{suffix}")


def _existing_files(root: Path, files: Iterable[str]) -> list[str]:
    return [path for path in files if (root / path).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="origin/main")
    args = parser.parse_args()

    root = Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), check=True).stdout.strip())
    ruff = shutil.which("ruff")
    mypy = shutil.which("mypy")
    if ruff is None or mypy is None:
        parser.error("ruff and mypy must be available on PATH")

    files = _changed_python_files(root, args.baseline)
    source_files = [path for path in files if path.startswith("src/")]
    if not files:
        print("No changed Python files.")
        return 0

    with tempfile.TemporaryDirectory(prefix="parse-bench-static-baseline-") as temp_dir:
        baseline_root = Path(temp_dir)
        _extract_baseline(root, args.baseline, baseline_root)
        baseline_files = _existing_files(baseline_root, files)
        baseline_source_files = _existing_files(baseline_root, source_files)

        baseline_ruff_list = _ruff_diagnostics(baseline_root, baseline_files, ruff)
        current_ruff_list = _ruff_diagnostics(root, files, ruff)
        baseline_ruff = Counter(baseline_ruff_list)
        current_ruff = Counter(current_ruff_list)
        new_ruff = _new_diagnostics(current_ruff, baseline_ruff)

        baseline_mypy_list = _mypy_diagnostics(baseline_root, baseline_source_files, mypy)
        current_mypy_list = _mypy_diagnostics(root, source_files, mypy)
        baseline_mypy = Counter(baseline_mypy_list)
        current_mypy = Counter(current_mypy_list)
        new_mypy = _new_diagnostics(current_mypy, baseline_mypy)

    changed_ranges = _added_line_ranges(root, args.baseline, files)
    touched_ruff = _on_changed_lines(current_ruff_list, changed_ranges)
    touched_mypy = _on_changed_lines(current_mypy_list, changed_ranges)
    format_failures = [
        (path, formatter_range)
        for path, formatter_range in _format_hunks(root, files, ruff)
        if any(_ranges_overlap(formatter_range, changed) for changed in changed_ranges.get(path, []))
    ]

    print(
        f"Checked {len(files)} changed Python files against {args.baseline}: "
        f"Ruff {sum(current_ruff.values())} current/{sum(baseline_ruff.values())} baseline, "
        f"mypy {sum(current_mypy.values())} current/{sum(baseline_mypy.values())} baseline."
    )
    if new_ruff:
        _print_diagnostics("new Ruff diagnostic", new_ruff)
    if new_mypy:
        _print_diagnostics("new mypy diagnostic", new_mypy)
    if touched_ruff:
        _print_diagnostics("Ruff diagnostic on branch-touched line", touched_ruff)
    if touched_mypy:
        _print_diagnostics("mypy diagnostic on branch-touched line", touched_mypy)
    for path, formatter_range in format_failures:
        print(f"Ruff format changes branch-touched lines: {path}:{formatter_range.start}-{formatter_range.stop - 1}")

    if new_ruff or new_mypy or touched_ruff or touched_mypy or format_failures:
        return 1
    print("Static quality delta passed: no new Ruff, Ruff-format, or mypy issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
