#!/usr/bin/env python3
"""Combine per-cell preprocessing reports into a single HTML with dropdown selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="*",
        help="Explicit list of per-cell feature reports (HTML) to include.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("."),
        help="Directory containing per-cell feature reports (defaults to current working directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("CellReports_15_S1_feature_report_all.html"),
        help="Path for the combined HTML output.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def discover_reports(base_dir: Path) -> List[Path]:
    reports: List[Path] = []
    if not base_dir.exists():
        return reports

    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            candidate = child / "CellReports_15_S1_feature_report.html"
            if candidate.exists():
                reports.append(candidate)
        elif child.is_file() and child.name.endswith("_feature_report.html"):
            reports.append(child)
    return reports


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def load_report_title(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        marker = "Viavi Cell"
        idx = text.find(marker)
        if idx != -1:
            start = text.find("<h1", max(0, idx - 200))
            end = text.find("</h1>", idx)
            if start != -1 and end != -1:
                return text[start:end].split(">")[1]
    except Exception:
        pass
    return path.parent.name


def wrap_report(content: str, cell_name: str, visible: bool) -> str:
    display = "block" if visible else "none"
    cid = sanitize_name(cell_name)
    return f"<div class='cell-report' id='cell-{cid}' style='display:{display}'>{content}</div>"


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    reports = list(dict.fromkeys(args.reports or []))  # deduplicate while preserving order
    if not reports:
        reports = discover_reports(args.features_dir)

    if not reports:
        raise FileNotFoundError("No preprocessing reports found; specify --reports or ensure per-cell reports exist.")

    sections: List[str] = []
    options: List[str] = []

    for idx, report_path in enumerate(reports):
        content = report_path.read_text(encoding="utf-8", errors="ignore")
        cell_name = load_report_title(report_path)
        cid = sanitize_name(cell_name)
        options.append(f"<option value='{cid}'>{cell_name}</option>")
        sections.append(wrap_report(content, cell_name, visible=(idx == 0)))

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>Cell Feature Engineering Reports</title>
      <style>
        body {{ font-family: Arial, sans-serif; background: #f7fafc; color: #243b53; margin: 2rem; }}
        select {{ padding: 0.6rem; font-size: 1rem; margin-bottom: 1.5rem; }}
        .cell-report {{ background: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
        iframe {{ width: 100%; border: none; min-height: 800px; }}
      </style>
      <script>
        function onCellChange(sel) {{
          document.querySelectorAll('.cell-report').forEach(sec => sec.style.display = 'none');
          const target = document.getElementById('cell-' + sel.value);
          if (target) target.style.display = 'block';
        }}
      </script>
    </head>
    <body>
      <h1>Cell Feature Engineering Reports</h1>
      <label for="cell-select">Select Cell:</label>
      <select id="cell-select" onchange="onCellChange(this)">
        {''.join(options)}
      </select>
      {''.join(sections)}
    </body>
    </html>
    """

    args.output.write_text(html, encoding="utf-8")
    print(f"[INFO] Combined report saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
