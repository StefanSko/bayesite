#!/usr/bin/env python3
"""Small deterministic HTML and inline-SVG helpers for statistical gates."""

from __future__ import annotations

import html
from typing import Iterable, Sequence

INK = "#0b0b0b"
BLUE = "#2a78d6"
RED = "#e34948"
GRID = "#e1e0d9"

_STYLE = """
:root { color-scheme: light; }
body { margin: 0 auto; max-width: 1180px; padding: 32px; background: white; color: #0b0b0b; font-family: system-ui, sans-serif; }
h1, h2 { line-height: 1.2; }
h2 { margin-top: 32px; }
.banner { display: inline-block; margin: 4px 0 20px; padding: 7px 16px; border-radius: 4px; color: white; font-weight: 700; letter-spacing: .08em; }
.banner.pass { background: #2a78d6; }
.banner.fail { background: #e34948; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 9px; border-bottom: 1px solid #e1e0d9; text-align: left; vertical-align: top; }
th { font-weight: 650; }
.settings { width: auto; min-width: 420px; }
.fail-row { background: #fde9e8; }
.figure { margin: 16px 0 28px; overflow-x: auto; }
svg { display: block; max-width: 100%; height: auto; background: white; }
.note { color: #555; }
""".strip()


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def header_block(
    title: str, settings: Sequence[tuple[str, object]], passed: bool
) -> str:
    """Render the report title, PASS/FAIL banner, and settings table."""
    rows = [([key, value], False) for key, value in settings]
    banner = "PASS" if passed else "FAIL"
    banner_class = "pass" if passed else "fail"
    return (
        f"<h1>{_escape(title)}</h1>\n"
        f'<div class="banner {banner_class}">{banner}</div>\n'
        + data_table(("Setting", "Value"), rows, css_class="settings")
    )


def section_heading(title: str) -> str:
    return f"<h2>{_escape(title)}</h2>"


def data_table(
    headers: Sequence[object],
    rows: Sequence[tuple[Sequence[object], bool]],
    *,
    css_class: str = "",
) -> str:
    """Render escaped tabular data; flagged rows receive ``fail-row`` styling."""
    class_attr = f' class="{_escape(css_class)}"' if css_class else ""
    head = "".join(f"<th>{_escape(value)}</th>" for value in headers)
    body: list[str] = []
    for values, failed in rows:
        row_class = ' class="fail-row"' if failed else ""
        cells = "".join(f"<td>{_escape(value)}</td>" for value in values)
        body.append(f"<tr{row_class}>{cells}</tr>")
    return (
        f"<table{class_attr}>\n<thead><tr>{head}</tr></thead>\n<tbody>\n"
        + "\n".join(body)
        + "\n</tbody>\n</table>"
    )


def html_document(
    title: str,
    settings: Sequence[tuple[str, object]],
    passed: bool,
    sections: Iterable[str],
) -> str:
    """Build a complete self-contained deterministic HTML document."""
    body = "\n".join([header_block(title, settings, passed), *sections])
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_escape(title)}</title>\n<style>\n{_STYLE}\n</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def svg_figure(width: int, height: int, label: str, elements: Iterable[str]) -> str:
    content = "\n".join(elements)
    return (
        '<div class="figure">\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_escape(label)}">\n'
        f"<title>{_escape(label)}</title>\n{content}\n</svg>\n</div>"
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = INK,
    width: float = 1.0,
    dashed: bool = False,
) -> str:
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (
        f'<line x1="{_number(x1)}" y1="{_number(y1)}" x2="{_number(x2)}" '
        f'y2="{_number(y2)}" stroke="{stroke}" stroke-width="{_number(width)}"{dash}/>'
    )


def svg_axes(x1: float, y1: float, x2: float, y2: float) -> str:
    return svg_path(f"M {_number(x1)} {_number(y1)} H {_number(x2)} M {_number(x1)} {_number(y1)} V {_number(y2)}")


def svg_circle(
    x: float, y: float, radius: float, *, fill: str = BLUE, stroke: str = "none"
) -> str:
    return (
        f'<circle cx="{_number(x)}" cy="{_number(y)}" r="{_number(radius)}" '
        f'fill="{fill}" stroke="{stroke}"/>'
    )


def svg_diamond(
    x: float, y: float, radius: float, *, fill: str = INK
) -> str:
    points = " ".join(
        (
            f"{_number(x)},{_number(y - radius)}",
            f"{_number(x + radius)},{_number(y)}",
            f"{_number(x)},{_number(y + radius)}",
            f"{_number(x - radius)},{_number(y)}",
        )
    )
    return f'<polygon points="{points}" fill="{fill}"/>'


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    opacity: float | None = None,
) -> str:
    opacity_attr = "" if opacity is None else f' opacity="{_number(opacity)}"'
    return (
        f'<rect x="{_number(x)}" y="{_number(y)}" width="{_number(width)}" '
        f'height="{_number(height)}" fill="{fill}" stroke="{stroke}"{opacity_attr}/>'
    )


def svg_path(
    data: str,
    *,
    stroke: str = INK,
    fill: str = "none",
    width: float = 1.0,
    opacity: float | None = None,
) -> str:
    opacity_attr = "" if opacity is None else f' opacity="{_number(opacity)}"'
    return (
        f'<path d="{_escape(data)}" stroke="{stroke}" fill="{fill}" '
        f'stroke-width="{_number(width)}"{opacity_attr}/>'
    )


def svg_text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 11,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{_number(x)}" y="{_number(y)}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{_escape(value)}</text>'
    )
