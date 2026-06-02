# HotPath Rules

## KP001: DataLoader Worker Count

Flags `DataLoader` calls with `num_workers=0` or no `num_workers`.

## KP002: Pinned Memory

Flags `DataLoader` calls missing `pin_memory=True`.

## KP003: `.item()` In Hot Loop

Flags scalar extraction inside loops because it can synchronize CPU and GPU.

## KP004: Device Transfer In Hot Loop

Flags `.to(...)`, `.cuda()`, and `.cpu()` inside loops.

## KP005: CPU Tensor Creation In Hot Loop

Flags tensor creation inside loops without `device=...`.

## KP006: Manual Attention Pattern

Flags files that combine matmul, transpose, and softmax patterns.

## KP007: Missing `torch.compile`

Flags training-like files that call backward but never call `torch.compile`.

## KP008: Missing Mixed Precision

Flags training-like files that call backward without autocast.

## KP009: Python Loop Over Tensor-Like Data

Flags loops over variables such as loaders, batches, and tensors.

## KP010: `zero_grad` Missing `set_to_none=True`

Flags optimizer zeroing that may perform unnecessary memory writes.

## KP011: `empty_cache` In Hot Loop

Flags `torch.cuda.empty_cache()` inside loops.

## KP012: Eval Without Inference Context

Flags files that call `eval()` without `no_grad()` or `inference_mode()`.

## KP013: `torch.tensor(existing_value)`

Flags copies through `torch.tensor(x)`.

## KP014: Missing Persistent Workers

Flags multi-worker DataLoaders missing `persistent_workers=True`.

## KP015: Missing `non_blocking=True`

Flags loop device transfers that omit `non_blocking=True`.

## KP016: Checkpoint Save In Hot Loop

Flags `torch.save(...)` inside loops.

## KP017: NumPy Conversion In Hot Loop

Flags `.numpy()` inside loops.
