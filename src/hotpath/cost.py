"""Impact and cost framing for HotPath findings.

This module is what makes HotPath different from a generic linter. Other tools
tell you *what* is wrong. HotPath also tells you, in plain terms, how much it is
likely *costing you* and which findings are worth your attention first.

Honesty note: static analysis cannot measure a real training run. We never
pretend to. Every rule carries a coarse impact tier, and the aggregate dollar
figure is an explicitly-labelled illustration based on assumptions you control
(`--gpu-hourly`, `--run-hours`, `--gpus`). The real number comes from a profiler
run -- that is the `hotpath profile` milestone on the roadmap.
"""

from dataclasses import replace

from hotpath.models import Finding

# ---------------------------------------------------------------------------
# Per-rule metadata
# ---------------------------------------------------------------------------
# confidence: how trustworthy the detection is with static info only.
#   "high"      -> precise AST match, very low false-positive rate
#   "medium"    -> inferential, usually right, occasional false positive
#   "heuristic" -> broad pattern, fires on legitimate code; hidden by default
#
# impact: rough magnitude of GPU time / cost when the issue is real.
#   "high"   -> can move training throughput by a large margin (~10-30%+)
#   "medium" -> a real but smaller tax (~1-10%)
#   "low"    -> good hygiene, marginal cost on its own

CONFIDENCE: dict[str, str] = {
    "KP001": "high",      # DataLoader num_workers
    "KP002": "high",      # pin_memory
    "KP003": "high",      # .item() in loop
    "KP004": "high",      # device transfer in loop
    "KP005": "high",      # cpu tensor creation in loop
    "KP006": "medium",    # manual attention pattern (co-occurrence)
    "KP007": "medium",    # missing torch.compile (file-level inference)
    "KP008": "medium",    # missing autocast (file-level inference)
    "KP009": "heuristic",  # python loop may block vectorization (very broad)
    "KP010": "high",      # zero_grad set_to_none
    "KP011": "high",      # empty_cache in loop
    "KP012": "heuristic",  # eval without no_grad (file-level, many FPs)
    "KP013": "high",      # torch.tensor copy
    "KP014": "high",      # persistent_workers
    "KP015": "high",      # non_blocking
    "KP016": "high",      # save in loop
    "KP017": "high",      # numpy in loop
}

IMPACT: dict[str, str] = {
    "KP001": "high",      # input pipeline stalls can bottleneck the whole run
    "KP002": "medium",
    "KP003": "high",      # GPU sync in the hot loop
    "KP004": "high",      # repeated transfers / sync in the hot loop
    "KP005": "medium",
    "KP006": "high",      # naive attention vs fused/Flash kernels
    "KP007": "high",      # torch.compile commonly 1.2-2x
    "KP008": "high",      # mixed precision commonly 1.3-2x + memory
    "KP009": "low",
    "KP010": "low",       # set_to_none is real but small
    "KP011": "medium",
    "KP012": "low",
    "KP013": "low",
    "KP014": "medium",
    "KP015": "medium",
    "KP016": "medium",
    "KP017": "medium",
}

# Short, money-oriented reason shown next to each finding.
COST_NOTE: dict[str, str] = {
    "KP001": "data loading can starve the GPU, leaving paid silicon idle",
    "KP002": "un-pinned memory slows every host-to-GPU batch copy",
    "KP003": "forces the CPU to wait for the GPU on every iteration",
    "KP004": "adds transfer + sync overhead on every iteration",
    "KP005": "allocates on CPU then pays for a copy to the GPU",
    "KP006": "hand-rolled attention misses fused/Flash kernels",
    "KP007": "skips graph compilation that often gives a large free speedup",
    "KP008": "full precision wastes throughput and GPU memory",
    "KP009": "per-item Python work can leave the GPU underutilized",
    "KP010": "extra memory writes when zeroing gradients",
    "KP011": "empty_cache in the loop stalls training for no real benefit",
    "KP012": "builds unnecessary autograd graphs during evaluation",
    "KP013": "an avoidable tensor copy",
    "KP014": "worker processes restart every epoch",
    "KP015": "blocking copies that can't overlap with compute",
    "KP016": "disk I/O on the hot path stalls the GPU",
    "KP017": "forces a CPU sync / device copy mid-loop",
}

# Illustrative throughput-recovery bands per dominant impact tier. Deliberately
# conservative and presented as a range, never a point estimate.
_BANDS = {
    "high": (0.10, 0.30),
    "medium": (0.02, 0.10),
    "low": (0.00, 0.02),
}

_IMPACT_RANK = {"high": 2, "medium": 1, "low": 0}


def annotate(finding: Finding) -> Finding:
    """Attach confidence and impact to a finding based on its rule id."""
    return replace(
        finding,
        confidence=CONFIDENCE.get(finding.rule_id, finding.confidence),
        impact=IMPACT.get(finding.rule_id, finding.impact),
    )


def cost_note(rule_id: str) -> str | None:
    return COST_NOTE.get(rule_id)


def dominant_impact(findings: list[Finding]) -> str:
    """The highest impact tier present among visible findings."""
    best = "low"
    for f in findings:
        if _IMPACT_RANK.get(f.impact, 0) > _IMPACT_RANK.get(best, 0):
            best = f.impact
    return best


def estimate_summary(
    findings: list[Finding],
    gpu_hourly: float = 2.50,
    run_hours: float = 12.0,
    gpus: int = 1,
) -> str:
    """A short, honest, dollar-anchored summary for a set of findings.

    The aggregate is intentionally an illustration: it anchors on a single
    conservative slowdown band for the highest-impact issue present, rather than
    summing rules (which would double-count and overclaim).
    """
    if not findings:
        return ""

    counts = {"high": 0, "medium": 0, "low": 0}
    for f in findings:
        counts[f.impact] = counts.get(f.impact, 0) + 1

    run_cost = gpu_hourly * run_hours * max(gpus, 1)
    tier = dominant_impact(findings)
    low_frac, high_frac = _BANDS[tier]
    low_dollars = run_cost * low_frac
    high_dollars = run_cost * high_frac

    lines = []
    lines.append(
        f"Impact: {counts['high']} high, {counts['medium']} medium, "
        f"{counts['low']} low."
    )
    lines.append(
        f"Reference run: {gpus}x GPU x {run_hours:g}h @ ${gpu_hourly:.2f}/h "
        f"= ${run_cost:,.0f}."
    )
    if tier == "low":
        lines.append(
            "No high-impact issues found. Remaining items are hygiene-level; "
            "savings are marginal on their own."
        )
    else:
        lines.append(
            f"Addressing the {tier}-impact findings could plausibly recover "
            f"~{low_frac*100:.0f}-{high_frac*100:.0f}% of throughput, on the "
            f"order of ${low_dollars:,.0f}-${high_dollars:,.0f} per run "
            f"(illustrative)."
        )
    lines.append(
        "Estimate only -- assumptions are yours to set; profile to confirm."
    )
    return "\n".join(lines)
