# HotPath Product Plan

HotPath is a GPU performance intelligence tool for PyTorch teams.

The first product is an open-source CLI:

```bash
hotpath analyze train.py
```

The long-term product is a hosted performance layer for AI teams:

- CI checks for expensive PyTorch regressions
- profiler trace analysis
- training-run monitoring
- team dashboards
- GPU cost and utilization reports
- eventually, verified code rewrites for safe optimizations

## Milestone 1: Open-Source CLI

Goal: 20-30 high-confidence static rules that developers trust.

Requirements:

- scan one file or a whole repository
- text and JSON output
- low false positive rate
- tests for every rule
- strong README with real examples

Success signal:

- HotPath finds real issues in 5 public PyTorch repos
- at least 10 developers run it on their own code
- at least 3 users say a finding was useful

## Milestone 2: Runtime Profiling

Goal: combine static linting with a short PyTorch profiler run.

Features:

- `hotpath profile train.py`
- detect DataLoader stalls
- detect CPU-GPU synchronization points
- identify top CUDA kernels by time
- export Chrome trace links
- explain profiler output in plain English

## Milestone 3: CI And Team Workflow

Goal: make HotPath useful before expensive training jobs launch.

Features:

- GitHub Action
- baseline comparison
- fail PRs on new high-confidence performance issues
- per-repo config file
- suppressions for accepted findings

## Milestone 4: Hosted Dashboard

Goal: turn the CLI into a company.

Features:

- training-run timeline
- GPU utilization summary
- throughput regression alerts
- team findings dashboard
- cost estimate per run
- recommendations ranked by likely savings

## Positioning

Raw profilers show traces. HotPath explains what is wrong and what to fix.

HotPath should be the first tool an AI developer runs before spending real GPU
money.
