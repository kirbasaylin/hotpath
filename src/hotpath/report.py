from pathlib import Path

from hotpath.cost import cost_note, estimate_summary
from hotpath.models import Finding

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}


def _visible(
    findings: list[Finding],
    min_severity: str,
    include_heuristics: bool,
) -> list[Finding]:
    threshold = SEVERITY_ORDER[min_severity]
    out = []
    for f in findings:
        if SEVERITY_ORDER[f.severity] < threshold:
            continue
        if f.confidence == "heuristic" and not include_heuristics:
            continue
        out.append(f)
    return out


def _impact_tag(finding: Finding) -> str:
    return {"high": "$$$", "medium": "$$", "low": "$"}.get(finding.impact, "$")


def render_findings(
    path: Path,
    findings: list[Finding],
    min_severity: str = "info",
    include_heuristics: bool = False,
    gpu_hourly: float = 2.50,
    run_hours: float = 12.0,
    gpus: int = 1,
) -> str:
    visible = _visible(findings, min_severity, include_heuristics)
    hidden = len(findings) - len(visible)

    lines = [f"HotPath analysis: {path}", ""]
    if not visible:
        lines.append("No findings at the selected severity.")
        return "\n".join(lines)

    for index, finding in enumerate(visible, start=1):
        lines.append(
            f"{index}. [{finding.severity.upper()}] {_impact_tag(finding)} "
            f"{finding.title} ({finding.rule_id})"
        )
        note = cost_note(finding.rule_id)
        if note:
            lines.append(f"   Why it costs: {note}.")
        lines.append(f"   Line {finding.line}: {finding.message}")
        lines.append(f"   Fix: {finding.recommendation}")
        if finding.example:
            lines.append("   Example:")
            for example_line in finding.example.splitlines():
                lines.append(f"     {example_line}")
        lines.append("")

    lines.append(estimate_summary(visible, gpu_hourly, run_hours, gpus))
    if hidden:
        lines.append(
            f"({hidden} low-confidence finding(s) hidden -- "
            f"pass --include-heuristics to show.)"
        )
    return "\n".join(lines).rstrip()


def render_project_findings(
    results: dict[Path, list[Finding]],
    min_severity: str = "info",
    include_heuristics: bool = False,
    gpu_hourly: float = 2.50,
    run_hours: float = 12.0,
    gpus: int = 1,
) -> str:
    files_with_visible = {
        path: _visible(findings, min_severity, include_heuristics)
        for path, findings in results.items()
    }
    files_with_visible = {
        path: findings for path, findings in files_with_visible.items() if findings
    }

    total = sum(len(findings) for findings in files_with_visible.values())
    all_visible = [f for findings in files_with_visible.values() for f in findings]
    lines = [
        f"HotPath project analysis: {total} findings across "
        f"{len(files_with_visible)} files",
        "",
    ]
    if not files_with_visible:
        lines.append("No findings at the selected severity.")
        return "\n".join(lines)

    for path, findings in sorted(files_with_visible.items()):
        lines.append(str(path))
        for finding in findings:
            lines.append(
                f"  [{finding.severity.upper()}] {_impact_tag(finding)} "
                f"{finding.rule_id} line {finding.line}: {finding.title}"
            )
            lines.append(f"    {finding.recommendation}")
        lines.append("")

    lines.append(estimate_summary(all_visible, gpu_hourly, run_hours, gpus))
    return "\n".join(lines).rstrip()
