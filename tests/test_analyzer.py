import unittest

from hotpath.analyzer import analyze_source


def rule_ids(source: str) -> set[str]:
    return {finding.rule_id for finding in analyze_source(source)}


class AnalyzerTests(unittest.TestCase):
    def test_detects_dataloader_worker_and_pin_memory_issues(self) -> None:
        source = """
import torch
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, num_workers=0)
"""
        ids = rule_ids(source)
        self.assertIn("KP001", ids)
        self.assertIn("KP002", ids)


    def test_detects_loop_sync_and_device_transfer(self) -> None:
        source = """
import torch

for batch in loader:
    x = batch.cuda()
    z = batch["z"].to(device)
    y = torch.zeros((2, 2))
    print(y.item())
"""
        ids = rule_ids(source)
        self.assertIn("KP003", ids)
        self.assertIn("KP004", ids)
        self.assertIn("KP005", ids)


    def test_detects_manual_attention(self) -> None:
        source = """
import torch

scores = torch.matmul(q, k.transpose(-2, -1))
probs = torch.softmax(scores, dim=-1)
out = torch.matmul(probs, v)
"""
        self.assertIn("KP006", rule_ids(source))


    def test_detects_training_without_compile_or_autocast(self) -> None:
        source = """
import torch

loss = model(x).mean()
loss.backward()
"""
        ids = rule_ids(source)
        self.assertIn("KP007", ids)
        self.assertIn("KP008", ids)


    def test_no_syntax_crash(self) -> None:
        findings = analyze_source("def broken(:")
        self.assertEqual("syntax-error", findings[0].rule_id)

    def test_detects_zero_grad_without_set_to_none(self) -> None:
        source = """
import torch

for batch in loader:
    optimizer.zero_grad()
    loss = model(batch).mean()
    loss.backward()
"""
        self.assertIn("KP010", rule_ids(source))

    def test_allows_zero_grad_with_set_to_none(self) -> None:
        source = """
import torch

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch).mean()
    loss.backward()
"""
        self.assertNotIn("KP010", rule_ids(source))

    def test_detects_empty_cache_inside_loop(self) -> None:
        source = """
import torch

for batch in loader:
    torch.cuda.empty_cache()
"""
        self.assertIn("KP011", rule_ids(source))

    def test_detects_eval_without_no_grad(self) -> None:
        source = """
import torch

model.eval()
preds = model(inputs)
"""
        self.assertIn("KP012", rule_ids(source))

    def test_allows_eval_with_inference_mode(self) -> None:
        source = """
import torch

model.eval()
with torch.inference_mode():
    preds = model(inputs)
"""
        self.assertNotIn("KP012", rule_ids(source))

    def test_detects_torch_tensor_existing_value(self) -> None:
        source = """
import torch

y = torch.tensor(x)
"""
        self.assertIn("KP013", rule_ids(source))

    def test_detects_missing_persistent_workers(self) -> None:
        source = """
import torch
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, num_workers=4)
"""
        self.assertIn("KP014", rule_ids(source))

    def test_detects_missing_non_blocking_on_loop_transfer(self) -> None:
        source = """
import torch

for batch in loader:
    x = batch.to(device)
"""
        self.assertIn("KP015", rule_ids(source))

    def test_allows_non_blocking_loop_transfer(self) -> None:
        source = """
import torch

for batch in loader:
    x = batch.to(device, non_blocking=True)
"""
        ids = rule_ids(source)
        self.assertNotIn("KP004", ids)
        self.assertNotIn("KP015", ids)

    def test_detects_torch_save_inside_loop(self) -> None:
        source = """
import torch

for step in range(10):
    torch.save(model.state_dict(), "model.pt")
"""
        self.assertIn("KP016", rule_ids(source))

    def test_detects_numpy_inside_loop(self) -> None:
        source = """
import torch

for batch in loader:
    values = batch.cpu().numpy()
"""
        self.assertIn("KP017", rule_ids(source))


if __name__ == "__main__":
    unittest.main()
