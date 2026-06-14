import unittest

from hotpath.analyzer import analyze_source
from hotpath.cost import dominant_impact, estimate_summary
from hotpath.report import render_findings, _visible
from pathlib import Path


TRAIN = """
import torch
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, num_workers=0)
for batch in loader:
    x = batch.to(device)
    loss = model(x).mean()
    loss.backward()
    print(loss.item())
"""


class CostAnnotationTests(unittest.TestCase):
    def test_findings_carry_impact_and_confidence(self) -> None:
        findings = analyze_source(TRAIN)
        self.assertTrue(findings)
        for f in findings:
            self.assertIn(f.impact, {"high", "medium", "low"})
            self.assertIn(f.confidence, {"high", "medium", "heuristic"})

    def test_dataloader_workers_is_high_impact(self) -> None:
        by_id = {f.rule_id: f for f in analyze_source(TRAIN)}
        self.assertEqual(by_id["KP001"].impact, "high")

    def test_dominant_impact_picks_highest(self) -> None:
        self.assertEqual(dominant_impact(analyze_source(TRAIN)), "high")

    def test_estimate_summary_scales_with_assumptions(self) -> None:
        findings = analyze_source(TRAIN)
        cheap = estimate_summary(findings, gpu_hourly=1.0, run_hours=1, gpus=1)
        pricey = estimate_summary(findings, gpu_hourly=4.0, run_hours=100, gpus=8)
        self.assertIn("Impact:", cheap)
        # A bigger, pricier run must produce a larger reference cost.
        self.assertNotEqual(cheap, pricey)

    def test_empty_findings_have_no_summary(self) -> None:
        self.assertEqual(estimate_summary([]), "")


class HeuristicGatingTests(unittest.TestCase):
    def test_range_loop_is_not_flagged_as_vectorization_issue(self) -> None:
        source = """
import torch
for epoch in range(10):
    train_one_epoch(model)
"""
        self.assertNotIn("KP009", {f.rule_id for f in analyze_source(source)})

    def test_heuristics_hidden_by_default_shown_on_request(self) -> None:
        source = """
import torch
model.eval()
out = model(x)
"""
        findings = analyze_source(source)
        # KP012 is heuristic; it exists in the raw findings...
        self.assertIn("KP012", {f.rule_id for f in findings})
        # ...but is filtered out of the default view...
        default_view = _visible(findings, "info", include_heuristics=False)
        self.assertNotIn("KP012", {f.rule_id for f in default_view})
        # ...and reappears when explicitly requested.
        full_view = _visible(findings, "info", include_heuristics=True)
        self.assertIn("KP012", {f.rule_id for f in full_view})

    def test_render_hides_heuristics_by_default(self) -> None:
        findings = analyze_source("import torch\nmodel.eval()\nout = model(x)\n")
        text = render_findings(Path("x.py"), findings)
        self.assertNotIn("KP012", text)


if __name__ == "__main__":
    unittest.main()
