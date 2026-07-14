#!/usr/bin/env python3
"""Stdlib self-tests for the conformance summary renderer."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "conformance_summary.py"
SPEC = importlib.util.spec_from_file_location("conformance_summary", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise AssertionError(f"could not load {SCRIPT}")
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


class ConformanceSummaryTests(unittest.TestCase):
    def test_stage_parsing(self) -> None:
        self.assertEqual(
            summary.parse_stage("NUTS oracle=success"),
            summary.Stage("NUTS oracle", "success"),
        )
        with self.assertRaisesRegex(summary.ConformanceError, "Display Name=result"):
            summary.parse_stage("missing-result")
        with self.assertRaisesRegex(summary.ConformanceError, "unsupported result"):
            summary.parse_stage("NUTS oracle=unknown")

    def test_exit_code_requires_every_stage_to_succeed(self) -> None:
        self.assertEqual(
            summary.exit_code([summary.Stage("A", "success")]), 0
        )
        self.assertEqual(
            summary.exit_code(
                [summary.Stage("A", "success"), summary.Stage("B", "failure")]
            ),
            1,
        )
        self.assertEqual(summary.exit_code([]), 1)

    def test_markdown_and_html_are_deterministic(self) -> None:
        stages = [
            summary.Stage("A", "success"),
            summary.Stage("B", "skipped"),
            summary.Stage("C", "cancelled"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            (reports / "z-gate").mkdir(parents=True)
            (reports / "a-gate").mkdir()
            (reports / "z-gate" / "z.html").write_text("z", encoding="utf-8")
            (reports / "a-gate" / "a.html").write_text("a", encoding="utf-8")
            first = summary.render_html(stages, reports)
            second = summary.render_html(stages, reports)

        self.assertEqual(first.encode(), second.encode())
        self.assertIn("Conformance summary", first)
        self.assertIn('<div class="banner fail">FAIL</div>', first)
        self.assertLess(first.index("a-gate/a.html"), first.index("z-gate/z.html"))
        markdown = summary.render_markdown(stages)
        self.assertIn("| A | PASS |", markdown)
        self.assertIn("| B | SKIPPED |", markdown)
        self.assertIn("| C | CANCELLED |", markdown)
        self.assertIn("Overall conformance verdict: **FAIL**", markdown)

    def test_main_writes_outputs_and_returns_stage_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "summary.html"
            step_summary = root / "step.md"
            code = summary.main(
                [
                    "--stage",
                    "A=success",
                    "--stage",
                    "B=success",
                    "--reports-dir",
                    str(root / "missing"),
                    "--report",
                    str(report),
                    "--step-summary",
                    str(step_summary),
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("(none found)", report.read_text(encoding="utf-8"))
            self.assertIn("**PASS**", step_summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
