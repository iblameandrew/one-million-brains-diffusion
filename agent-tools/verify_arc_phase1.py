#!/usr/bin/env python3
"""Local verification for ARC Phase-1 logging and generate path (no GPU required)."""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tfbd.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")


class TestSourceStructure(unittest.TestCase):
    def test_version_is_tfbd_v(self):
        self.assertIn('SCRIPT_VERSION = "2026-06-20-tfbd-v"', SOURCE)

    def test_no_quantized_loader(self):
        self.assertNotIn("BitsAndBytesConfig", SOURCE)
        self.assertNotIn("_load_diffusiongemma_bnb", SOURCE)
        self.assertNotIn("strategy 0: bitsandbytes", SOURCE)

    def test_tfbd_orchestrator_present(self):
        self.assertIn("class TFBD_Orchestrator", SOURCE)
        self.assertIn("class TopologicalFiberEmbedding", SOURCE)
        self.assertIn("class CohomologicalStitcher", SOURCE)
        self.assertIn("def tfbd_generate(", SOURCE)

    def test_hf_only_no_vllm_import(self):
        self.assertNotIn("from vllm import", SOURCE)
        self.assertNotIn("ARC_TRY_VLLM", SOURCE)

    def test_single_engine_arc_no_voter_pool(self):
        self.assertNotIn("ARC_MULTI_AGENT_REQUIRED", SOURCE)
        self.assertNotIn("MultiAgentEnginePool", SOURCE)
        self.assertNotIn("def _resolve_voter_pool_gen_path()", SOURCE)
        self.assertNotIn("MBR_AGENT_WORKER", SOURCE)

    def test_spatial_thinking_disabled(self):
        self.assertIn("ARC_SPATIAL_ENABLE_THINKING = False", SOURCE)

    def test_spatial_ensemble_enabled(self):
        self.assertIn("ARC_SPATIAL_GRID_ENSEMBLE = True", SOURCE)
        self.assertIn("arc_spatial_grid_ensemble_pipeline(", SOURCE)

    def test_phase2_tfbd_stitch_and_pixel_fallback(self):
        self.assertIn("ENABLE_TFBD", SOURCE)
        self.assertIn("orch.stitcher.stitch(", SOURCE)
        self.assertIn("pixel_wise_majority_vote_grids(", SOURCE)

    def test_phase1_uses_engine_generate(self):
        self.assertIn("def _engine_generate_arc(", SOURCE)
        self.assertIn("_arc_phase1_generate_slots(", SOURCE)

    def test_phase1_parallelism_enabled(self):
        self.assertIn("ARC_PHASE1_PROMPT_PARALLELISM = True", SOURCE)

    def test_ast_parses(self):
        ast.parse(SOURCE)


class TestArcBudgetHelpers(unittest.TestCase):
    """Replicate budget helpers from script constants."""

    ARC_MBR_OUTPUT_TOKEN_BUDGET = 14000
    ARC_FINAL_GRID_MIN_TOKENS = 512
    ARC_FINAL_GRID_MAX_FRACTION = 0.85
    ARC_FINAL_GRID_MIN_FRACTION = 0.50
    ARC_HYPOTHESIS_SLOTS = 8

    def _estimate_grid_json_tokens(self, grid):
        import json

        return max(32, len(json.dumps(grid)) // 2)

    def _arc_final_grid_max_tokens(self, task):
        budget = int(self.ARC_MBR_OUTPUT_TOKEN_BUDGET)
        floor = max(int(self.ARC_FINAL_GRID_MIN_TOKENS), budget // 4)
        ceiling = max(floor, int(budget * self.ARC_FINAL_GRID_MAX_FRACTION))
        need = floor
        for ex in task.get("train", []):
            out = ex.get("output") or []
            if out:
                need = max(need, self._estimate_grid_json_tokens(out))
        return min(ceiling, max(floor, need))

    def _arc_spatial_slot_max_tokens(self, task):
        est = 256
        for ex in task.get("train", []):
            for key in ("input", "output"):
                grid = ex.get(key) or []
                if grid:
                    est = max(est, self._estimate_grid_json_tokens(grid))
        est = est + 128
        budget = int(self.ARC_MBR_OUTPUT_TOKEN_BUDGET)
        per_slot_share = max(256, budget // max(1, self.ARC_HYPOTHESIS_SLOTS))
        return max(64, min(est, per_slot_share))

    def test_spatial_per_slot_budget_within_share(self):
        task = {
            "train": [
                {"input": [[0]], "output": [[0] * 30 for _ in range(30)]},
            ],
            "test": [{"input": [[0] * 30 for _ in range(30)]}],
        }
        per_slot = self._arc_spatial_slot_max_tokens(task)
        share = max(256, self.ARC_MBR_OUTPUT_TOKEN_BUDGET // self.ARC_HYPOTHESIS_SLOTS)
        self.assertLessEqual(per_slot, share)
        self.assertGreaterEqual(per_slot, 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)