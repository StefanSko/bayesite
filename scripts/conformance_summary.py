#!/usr/bin/env python3
"""Render a deterministic summary of conformance CI stage results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gate_report as report_html

_ALLOWED_RESULTS = {"success", "failure", "cancelled", "skipped"}
_MARKDOWN_RESULTS = {
    "success": "PASS",
    "failure": "FAIL",
    "cancelled": "CANCELLED",
    "skipped": "SKIPPED",
}


class ConformanceError(ValueError):
    """Invalid conformance summary input."""


@dataclass(frozen=True)
class Stage:
    name: str
    result: str


def parse_stage(value: str) -> Stage:
    name, separator, result = value.partition("=")
    name = name.strip()
    result = result.strip().lower()
    if not separator or not name:
        raise ConformanceError('--stage must use "Display Name=result"')
    if result not in _ALLOWED_RESULTS:
        allowed = ", ".join(sorted(_ALLOWED_RESULTS))
        raise ConformanceError(
            f"stage {name!r} has unsupported result {result!r}; expected one of {allowed}"
        )
    return Stage(name=name, result=result)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        metavar="DISPLAY=result",
        help="conformance stage and GitHub needs result",
    )
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--step-summary", type=Path)
    return parser.parse_args(argv)


def found_report_files(reports_dir: Path) -> tuple[str, ...]:
    if not reports_dir.is_dir():
        return ()
    return tuple(
        sorted(
            path.relative_to(reports_dir).as_posix()
            for path in reports_dir.rglob("*")
            if path.is_file()
        )
    )


def all_stages_passed(stages: Sequence[Stage]) -> bool:
    return bool(stages) and all(stage.result == "success" for stage in stages)


def exit_code(stages: Sequence[Stage]) -> int:
    return 0 if all_stages_passed(stages) else 1


def render_html(stages: Sequence[Stage], reports_dir: Path) -> str:
    passed = all_stages_passed(stages)
    settings = tuple((stage.name, stage.result) for stage in stages) + (
        ("overall", "PASS" if passed else "FAIL"),
    )
    report_files = found_report_files(reports_dir)
    rows = (
        [((relative_path,), False) for relative_path in report_files]
        if report_files
        else [(('(none found)',), False)]
    )
    sections = (
        report_html.section_heading("Per-gate reports found"),
        report_html.data_table(("Relative path",), rows),
    )
    return report_html.html_document(
        "Conformance summary", settings, passed, sections
    )


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(stages: Sequence[Stage]) -> str:
    lines = [
        "## Conformance summary",
        "",
        "| Stage | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| {_markdown_cell(stage.name)} | {_MARKDOWN_RESULTS[stage.result]} |"
        for stage in stages
    )
    verdict = "PASS" if all_stages_passed(stages) else "FAIL"
    lines.extend(("", f"Overall conformance verdict: **{verdict}**", ""))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        stages = [parse_stage(value) for value in args.stage]
    except ConformanceError as error:
        print(f"conformance summary failed: {error}", file=sys.stderr)
        return 2
    args.report.write_text(render_html(stages, args.reports_dir), encoding="utf-8")
    if args.step_summary is not None:
        with args.step_summary.open("a", encoding="utf-8") as summary:
            summary.write(render_markdown(stages))
    return exit_code(stages)


if __name__ == "__main__":
    sys.exit(main())
