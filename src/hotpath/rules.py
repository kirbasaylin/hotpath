import ast

from hotpath.models import Finding


class PyTorchPerformanceVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self._findings: list[Finding] = []
        self._loop_depth = 0
        self._imports_torch = False
        self._uses_compile = False
        self._uses_autocast = False
        self._uses_backward = False
        self._uses_no_grad = False
        self._uses_eval = False
        self._saw_matmul = False
        self._saw_softmax = False
        self._saw_transpose = False

    def findings(self) -> list[Finding]:
        final = list(self._findings)
        if self._imports_torch and self._uses_backward and not self._uses_compile:
            final.append(
                Finding(
                    rule_id="KP007",
                    title="Training script does not use torch.compile",
                    severity="info",
                    line=1,
                    message="This file appears to train a PyTorch model but never calls torch.compile.",
                    recommendation=(
                        "Try model = torch.compile(model) on PyTorch 2.x, then benchmark "
                        "a few warmup iterations before trusting the speedup."
                    ),
                    example="model = torch.compile(model)",
                )
            )
        if self._imports_torch and self._uses_backward and not self._uses_autocast:
            final.append(
                Finding(
                    rule_id="KP008",
                    title="Training loop does not appear to use mixed precision",
                    severity="info",
                    line=1,
                    message="No autocast context was found in a file that appears to train a model.",
                    recommendation=(
                        "On modern NVIDIA GPUs, try torch.autocast with bf16 or fp16 and "
                        "measure memory use and throughput."
                    ),
                    example=(
                        "with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16):\n"
                        "    loss = model(inputs)"
                    ),
                )
            )
        if self._saw_matmul and self._saw_softmax and self._saw_transpose:
            final.append(
                Finding(
                    rule_id="KP006",
                    title="Manual attention pattern detected",
                    severity="warning",
                    line=1,
                    message=(
                        "This file uses matmul, softmax, and transpose patterns that often "
                        "show up in hand-written attention."
                    ),
                    recommendation=(
                        "Consider torch.nn.functional.scaled_dot_product_attention, which can "
                        "dispatch to optimized kernels such as FlashAttention when supported."
                    ),
                    example=(
                        "attn = torch.nn.functional.scaled_dot_product_attention(\n"
                        "    query, key, value, is_causal=True\n"
                        ")"
                    ),
                )
            )
        if self._uses_eval and not self._uses_no_grad:
            final.append(
                Finding(
                    rule_id="KP012",
                    title="Evaluation path may be missing no_grad",
                    severity="warning",
                    line=1,
                    message="This file calls eval() but does not use torch.no_grad() or torch.inference_mode().",
                    recommendation=(
                        "Wrap inference or validation code in torch.no_grad() or torch.inference_mode() "
                        "to avoid building unnecessary autograd graphs."
                    ),
                    example=(
                        "model.eval()\n"
                        "with torch.inference_mode():\n"
                        "    outputs = model(inputs)"
                    ),
                )
            )
        return sorted(final, key=lambda finding: (finding.line, finding.rule_id))

    def visit_Import(self, node: ast.Import) -> None:
        if any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names):
            self._imports_torch = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and (node.module == "torch" or node.module.startswith("torch.")):
            self._imports_torch = True
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self._looks_like_tensor_loop(node.iter):
            self._add(
                "KP009",
                "Python loop may block vectorization",
                "info",
                node.lineno,
                "A Python for-loop appears to iterate over tensors, batches, or a DataLoader.",
                (
                    "Check whether this work can be batched or vectorized. Python loops can "
                    "leave the GPU underutilized when each iteration launches small kernels."
                ),
            )
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            context = item.context_expr
            name = dotted_name(context.func) if isinstance(context, ast.Call) else dotted_name(context)
            if name and (name.endswith("no_grad") or name.endswith("inference_mode")):
                self._uses_no_grad = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            name = dotted_name(decorator.func) if isinstance(decorator, ast.Call) else dotted_name(decorator)
            if name and (name.endswith("no_grad") or name.endswith("inference_mode")):
                self._uses_no_grad = True
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)

        if name in {"torch.compile", "compile"}:
            self._uses_compile = True
        if name and "autocast" in name:
            self._uses_autocast = True
        if name and name.endswith(".backward"):
            self._uses_backward = True
        if name and name.endswith(".eval"):
            self._uses_eval = True
        if name and ("matmul" in name or name.endswith(".bmm")):
            self._saw_matmul = True
        if name and "softmax" in name:
            self._saw_softmax = True
        if name and (name.endswith(".transpose") or name.endswith(".permute")):
            self._saw_transpose = True

        if name and (name == "item" or name.endswith(".item")) and self._loop_depth > 0:
            self._add(
                "KP003",
                ".item() inside loop can synchronize the GPU",
                "warning",
                node.lineno,
                ".item() copies a scalar to Python and can force the CPU to wait for GPU work.",
                "Accumulate tensors on-device and call .item() only for occasional logging.",
                "if step % log_every == 0:\n    print(loss.detach().item())",
            )

        if self._loop_depth > 0 and name and should_warn_device_transfer(name, node):
            self._add(
                "KP004",
                "Device transfer inside loop",
                "warning",
                node.lineno,
                f"{name} is called inside a loop, which can add transfer overhead or synchronization.",
                (
                    "Move tensors to the target device before the hot loop when possible. "
                    "For batches, use non_blocking=True with pinned memory."
                ),
                "inputs = inputs.to(device, non_blocking=True)",
            )
        if self._loop_depth > 0 and name and is_host_to_device_transfer(name) and not has_keyword(node, "non_blocking"):
            self._add(
                "KP015",
                "Batch transfer missing non_blocking=True",
                "info",
                node.lineno,
                f"{name} moves data inside a loop without non_blocking=True.",
                (
                    "When the DataLoader uses pin_memory=True, pass non_blocking=True "
                    "to overlap host-to-device copies with GPU work."
                ),
                "inputs = inputs.to(device, non_blocking=True)",
            )

        if self._loop_depth > 0 and name in CPU_TENSOR_CREATORS and not has_keyword(node, "device"):
            self._add(
                "KP005",
                "Tensor created on CPU inside loop",
                "warning",
                node.lineno,
                f"{name} creates a CPU tensor inside a loop and may require a later GPU copy.",
                "Create tensors directly on the target device with device=...",
                "mask = torch.zeros(shape, device=device)",
            )

        if name and name.endswith("DataLoader"):
            self._check_dataloader(node)

        if name and name.endswith(".zero_grad") and not has_keyword(node, "set_to_none"):
            self._add(
                "KP010",
                "zero_grad missing set_to_none=True",
                "info",
                node.lineno,
                "optimizer.zero_grad() clears gradients by writing zeros, which can add memory work.",
                (
                    "Try optimizer.zero_grad(set_to_none=True). It can reduce memory writes "
                    "and is the PyTorch-recommended faster path for many training loops."
                ),
                "optimizer.zero_grad(set_to_none=True)",
            )

        if self._loop_depth > 0 and name == "torch.cuda.empty_cache":
            self._add(
                "KP011",
                "empty_cache called inside loop",
                "warning",
                node.lineno,
                "torch.cuda.empty_cache() inside a hot loop can slow training and rarely fixes real leaks.",
                "Remove empty_cache() from the training loop. Use it only between large phases if needed.",
                "# Avoid calling torch.cuda.empty_cache() every iteration",
            )

        if name == "torch.tensor" and node.args and looks_like_existing_tensor(node.args[0]):
            self._add(
                "KP013",
                "torch.tensor called on an existing value",
                "warning",
                node.lineno,
                "torch.tensor(x) copies data and may detach gradients when x is already tensor-like.",
                "Use x.clone().detach(), torch.as_tensor(x), or create the tensor on the target device directly.",
                "x = existing.clone().detach()",
            )

        if self._loop_depth > 0 and name == "torch.save":
            self._add(
                "KP016",
                "Checkpoint save inside loop",
                "info",
                node.lineno,
                "torch.save inside a hot loop can stall training on disk I/O.",
                "Save checkpoints every N steps or at epoch boundaries instead of every iteration.",
                "if step % save_every == 0:\n    torch.save(state, path)",
            )

        if self._loop_depth > 0 and name and (name == "numpy" or name.endswith(".numpy")):
            self._add(
                "KP017",
                "NumPy conversion inside loop",
                "warning",
                node.lineno,
                ".numpy() inside a hot loop can force CPU synchronization or require a device copy.",
                "Keep metrics on tensors during the loop and convert to NumPy only outside the hot path.",
                "metric_values.append(metric.detach())",
            )

        self.generic_visit(node)

    def _check_dataloader(self, node: ast.Call) -> None:
        num_workers = keyword_value(node, "num_workers")
        if num_workers is None:
            self._add(
                "KP001",
                "DataLoader missing num_workers",
                "warning",
                node.lineno,
                "This DataLoader does not set num_workers, so data loading may run in the main process.",
                "Benchmark num_workers values such as 2, 4, or 8 for your machine and dataset.",
                "DataLoader(dataset, batch_size=64, num_workers=4)",
            )
        elif isinstance(num_workers, ast.Constant) and num_workers.value == 0:
            self._add(
                "KP001",
                "DataLoader uses num_workers=0",
                "warning",
                node.lineno,
                "num_workers=0 can starve the GPU when preprocessing or disk reads are non-trivial.",
                "Benchmark num_workers values such as 2, 4, or 8 for your machine and dataset.",
                "DataLoader(dataset, batch_size=64, num_workers=4)",
            )
        elif literal_int(num_workers) and literal_int(num_workers) > 0:
            persistent_workers = keyword_value(node, "persistent_workers")
            if persistent_workers is None or literal_bool(persistent_workers) is False:
                self._add(
                    "KP014",
                    "DataLoader workers are not persistent",
                    "info",
                    node.lineno,
                    "num_workers is greater than zero, but persistent_workers=True is not set.",
                    (
                        "For multi-epoch training, try persistent_workers=True to avoid "
                        "restarting worker processes each epoch."
                    ),
                    "DataLoader(dataset, num_workers=4, persistent_workers=True)",
                )

        pin_memory = keyword_value(node, "pin_memory")
        if pin_memory is None:
            self._add(
                "KP002",
                "DataLoader missing pin_memory=True",
                "info",
                node.lineno,
                "Pinned host memory can make CPU-to-GPU batch transfers faster.",
                "If training on CUDA, try pin_memory=True and batch.to(device, non_blocking=True).",
                "DataLoader(dataset, batch_size=64, pin_memory=True)",
            )
        elif isinstance(pin_memory, ast.Constant) and pin_memory.value is False:
            self._add(
                "KP002",
                "DataLoader uses pin_memory=False",
                "info",
                node.lineno,
                "pin_memory=False can slow host-to-GPU transfers for CUDA training.",
                "If training on CUDA, try pin_memory=True and batch.to(device, non_blocking=True).",
                "DataLoader(dataset, batch_size=64, pin_memory=True)",
            )

    def _looks_like_tensor_loop(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return any(token in node.id.lower() for token in LOOP_TOKENS)
        if isinstance(node, ast.Call):
            name = dotted_name(node.func)
            # Only data iteration (e.g. a DataLoader) is a vectorization smell.
            # Plain range() loops are the normal epoch/step counters and must
            # not be flagged, or trust in the rule collapses.
            return bool(name and name.endswith("DataLoader"))
        if isinstance(node, ast.Attribute):
            return any(token in node.attr.lower() for token in LOOP_TOKENS)
        return False

    def _add(
        self,
        rule_id: str,
        title: str,
        severity: str,
        line: int,
        message: str,
        recommendation: str,
        example: str | None = None,
    ) -> None:
        self._findings.append(
            Finding(
                rule_id=rule_id,
                title=title,
                severity=severity,
                line=line,
                message=message,
                recommendation=recommendation,
                example=example,
            )
        )


CPU_TENSOR_CREATORS = {
    "torch.tensor",
    "torch.zeros",
    "torch.ones",
    "torch.empty",
    "torch.arange",
    "torch.eye",
    "torch.randn",
    "torch.rand",
    "torch.full",
}

LOOP_TOKENS = {"loader", "dataloader", "batch", "batches", "tensor", "tensors"}


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def keyword_value(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def has_keyword(node: ast.Call, name: str) -> bool:
    return keyword_value(node, name) is not None


def is_device_transfer_call(name: str) -> bool:
    return name in {"cpu", "cuda", "to"} or (
        name.endswith(".cpu") or name.endswith(".cuda") or name.endswith(".to")
    )


def is_host_to_device_transfer(name: str) -> bool:
    return name in {"cuda", "to"} or name.endswith(".cuda") or name.endswith(".to")


def is_cpu_transfer(name: str) -> bool:
    return name == "cpu" or name.endswith(".cpu")


def should_warn_device_transfer(name: str, node: ast.Call) -> bool:
    if not is_device_transfer_call(name):
        return False
    if is_cpu_transfer(name):
        return True
    return not has_keyword(node, "non_blocking")


def literal_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def literal_bool(node: ast.AST | None) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def looks_like_existing_tensor(node: ast.AST) -> bool:
    return isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call))
