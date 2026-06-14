from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    rule_id: str
    title: str
    severity: str
    line: int
    message: str
    recommendation: str
    example: str | None = None
    # How likely this finding is a true positive given only static information.
    # One of: "high", "medium", "heuristic". Heuristic findings are hidden by
    # default so a noisy rule can never erode trust in the precise ones.
    confidence: str = "high"
    # Rough magnitude of GPU time / cost if the issue is real.
    # One of: "high", "medium", "low".
    impact: str = "medium"
