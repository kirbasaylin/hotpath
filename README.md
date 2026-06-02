# PerfForge

Find slow PyTorch code before it burns GPU hours.

PerfForge is a small rule-based CLI for detecting common PyTorch performance anti-patterns before expensive training runs, CI jobs, or experiments start.


> Package note: the PyPI name `hotpath` is used by an unrelated project.
> This project publishes as `hotpath-ai`, while the command remains `hotpath`.

## Why HotPath

GPU profilers are powerful, but they often show you traces after money has
already been spent. HotPath catches common PyTorch performance mistakes before
long training runs, CI jobs, or expensive experiments start.

Use it when you want to spot:

- CPU/GPU synchronization traps like `.item()` in hot loops
- slow input pipelines from underconfigured `DataLoader`s
- blocking device transfers
- missing mixed precision or `torch.compile`
- checkpointing and conversion work inside training loops

## Quick Start

```bash
python -m hotpath analyze examples/slow_training.py
```

Install from PyPI:

```bash
pip install hotpath-ai
hotpath analyze examples/slow_training.py
```

Or after installing locally from this repo:

```bash
pip install -e .
hotpath analyze examples/slow_training.py
```

Scan a whole project:

```bash
hotpath analyze path/to/project
```

Emit JSON for CI, scripts, or a future dashboard:

```bash
hotpath analyze path/to/project --format json
```

Fail CI when warnings are present:

```bash
hotpath analyze path/to/project --fail-on warning
```

## Current Checks

- `DataLoader` with `num_workers=0` or no `num_workers`
- `DataLoader` missing `pin_memory=True`
- `.item()` inside loops, which can force GPU synchronization
- `.cpu()`, `.cuda()`, or `.to(...)` calls inside loops
- CPU tensor creation inside loops without a `device=...`
- manual attention patterns that may be replaced with
  `torch.nn.functional.scaled_dot_product_attention`
- training scripts that never call `torch.compile`
- training scripts that appear to use full precision without autocast
- Python loops over tensors or batches that may block vectorization
- `optimizer.zero_grad()` missing `set_to_none=True`
- `torch.cuda.empty_cache()` inside hot loops
- evaluation code that calls `eval()` without `no_grad()` or `inference_mode()`
- `torch.tensor(existing_value)` copies
- `DataLoader` workers missing `persistent_workers=True`
- loop transfers missing `non_blocking=True`
- `torch.save(...)` inside hot loops
- `.numpy()` conversion inside hot loops

## Example Output

```text
[WARNING] KP003 line 31: .item() inside loop can synchronize the GPU
  Accumulate tensors on-device and call .item() only for occasional logging.

[INFO] KP010 line 29: zero_grad missing set_to_none=True
  Try optimizer.zero_grad(set_to_none=True).
```

## GitHub Topics

Recommended repository topics:

`pytorch`, `cuda`, `gpu`, `gpu-optimization`, `performance`, `profiling`,
`static-analysis`, `linting`, `machine-learning`, `deep-learning`, `mlops`,
`ai-tools`, `developer-tools`, `training`, `torch`, `triton`

## Philosophy

HotPath is intentionally not a magic kernel generator yet. The first product
is the GPU performance linter experts wish existed: clear detections,
plain-English explanations, and copy-pasteable fixes.

## PyPI Name

The installable package name is `hotpath-ai` because `hotpath` is already used
by an unrelated PyPI project. The Python import package and command-line entry
point are still `hotpath`:

```bash
pip install hotpath-ai
hotpath analyze path/to/project
```

## Roadmap

See [PRODUCT_PLAN.md](PRODUCT_PLAN.md).
