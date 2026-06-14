import ast
from pathlib import Path

from hotpath.cost import annotate
from hotpath.models import Finding
from hotpath.rules import PyTorchPerformanceVisitor


def analyze_source(source: str, filename: str = "<source>") -> list[Finding]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [
            Finding(
                rule_id="syntax-error",
                title="Python syntax error",
                severity="error",
                line=exc.lineno or 1,
                message=exc.msg,
                recommendation="Fix the syntax error before running HotPath again.",
            )
        ]

    visitor = PyTorchPerformanceVisitor(source)
    visitor.visit(tree)
    return [annotate(finding) for finding in visitor.findings()]


def analyze_file(path: Path) -> list[Finding]:
    if not path.exists():
        return [
            Finding(
                rule_id="file-not-found",
                title="File not found",
                severity="error",
                line=1,
                message=f"{path} does not exist.",
                recommendation="Pass a valid Python file path.",
            )
        ]

    source = path.read_text(encoding="utf-8")
    return analyze_source(source, filename=str(path))
