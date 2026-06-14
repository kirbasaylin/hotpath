import argparse
import json
from pathlib import Path

from hotpath.analyzer import analyze_file
from hotpath.report import render_findings, render_project_findings


DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hotpath",
        description="Find common PyTorch GPU performance anti-patterns.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a Python file or folder.")
    analyze.add_argument("path", type=Path, help="Path to a Python file or folder.")
    analyze.add_argument(
        "--min-severity",
        choices=["info", "warning", "error"],
        default="info",
        help="Only show findings at or above this severity.",
    )
    analyze.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    analyze.add_argument(
        "--fail-on",
        choices=["none", "warning", "error"],
        default="none",
        help="Exit with status 1 when findings at or above this severity are present.",
    )
    analyze.add_argument(
        "--include-heuristics",
        action="store_true",
        help="Show low-confidence (heuristic) findings that are hidden by default.",
    )
    analyze.add_argument(
        "--gpu-hourly",
        type=float,
        default=2.50,
        help="Assumed GPU price per hour for the cost estimate (default: 2.50).",
    )
    analyze.add_argument(
        "--run-hours",
        type=float,
        default=12.0,
        help="Assumed training run length in hours for the cost estimate (default: 12).",
    )
    analyze.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Assumed number of GPUs for the cost estimate (default: 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        results = analyze_path(args.path)
        if args.format == "json":
            print(json.dumps(json_payload(results, args), indent=2))
        elif len(results) == 1:
            path, findings = next(iter(results.items()))
            print(
                render_findings(
                    path,
                    findings,
                    min_severity=args.min_severity,
                    include_heuristics=args.include_heuristics,
                    gpu_hourly=args.gpu_hourly,
                    run_hours=args.run_hours,
                    gpus=args.gpus,
                )
            )
        else:
            print(
                render_project_findings(
                    results,
                    min_severity=args.min_severity,
                    include_heuristics=args.include_heuristics,
                    gpu_hourly=args.gpu_hourly,
                    run_hours=args.run_hours,
                    gpus=args.gpus,
                )
            )
        return exit_code(results, args.fail_on)

    parser.error(f"Unknown command: {args.command}")
    return 2


def analyze_path(path: Path) -> dict[Path, list]:
    if path.is_file():
        return {path: analyze_file(path)}
    if path.is_dir():
        results = {}
        for python_file in iter_python_files(path):
            findings = analyze_file(python_file)
            if findings:
                results[python_file] = findings
        return results
    return {path: analyze_file(path)}


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDES for part in path.parts):
            continue
        yield path


def json_payload(results: dict[Path, list], args=None) -> dict:
    from hotpath.cost import cost_note, estimate_summary

    gpu_hourly = getattr(args, "gpu_hourly", 2.50)
    run_hours = getattr(args, "run_hours", 12.0)
    gpus = getattr(args, "gpus", 1)

    all_findings = [f for findings in results.values() for f in findings]
    return {
        "files": [
            {
                "path": str(path),
                "findings": [
                    {
                        "rule_id": finding.rule_id,
                        "title": finding.title,
                        "severity": finding.severity,
                        "impact": finding.impact,
                        "confidence": finding.confidence,
                        "line": finding.line,
                        "message": finding.message,
                        "cost_note": cost_note(finding.rule_id),
                        "recommendation": finding.recommendation,
                        "example": finding.example,
                    }
                    for finding in findings
                ],
            }
            for path, findings in results.items()
        ],
        "cost_summary": estimate_summary(
            all_findings, gpu_hourly, run_hours, gpus
        ),
    }


def exit_code(results: dict[Path, list], fail_on: str) -> int:
    if fail_on == "none":
        return 0
    threshold = {"warning": 1, "error": 2}[fail_on]
    severity_value = {"info": 0, "warning": 1, "error": 2}
    for findings in results.values():
        if any(severity_value[finding.severity] >= threshold for finding in findings):
            return 1
    return 0
