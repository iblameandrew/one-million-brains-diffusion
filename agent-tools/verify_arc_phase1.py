#!/usr/bin/env python3
"""Local verification for ARC Phase-1 logging and generate path (no GPU required)."""
from __future__ import annotations

import ast
import re
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "million_brains_dflash.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")


class TestSourceStructure(unittest.TestCase):
    def test_version_is_17r(self):
        self.assertIn('SCRIPT_VERSION = "2026-06-17r"', SOURCE)

    def test_voter_pool_required_by_default(self):
        self.assertIn("ARC_MULTI_AGENT_REQUIRED = True", SOURCE)
        self.assertIn("def _materialize_worker_script_path()", SOURCE)
        self.assertIn("def _resolve_voter_pool_gen_path()", SOURCE)
        self.assertNotIn(
            "voter pool spawn failed — using single engine",
            SOURCE,
        )

    def test_hypothesis_thinking_disabled(self):
        self.assertIn("ARC_HYPOTHESIS_ENABLE_THINKING = False", SOURCE)

    def test_phase1_uses_use_tqdm_false(self):
        self.assertIn("outs = vllm_llm.generate(prompts, sp_list, use_tqdm=False)", SOURCE)

    def test_phase2_uses_use_tqdm_false(self):
        self.assertIn("out = vllm_llm.generate([prompt], sp, use_tqdm=False)[0]", SOURCE)

    def test_phase1_timing_logs(self):
        self.assertIn("[ARC-PHASE-1] Generate start:", SOURCE)
        self.assertIn("[ARC-PHASE-1] Generate done:", SOURCE)

    def test_no_slot_temp_before_loop_bug(self):
        # slot_temp must not appear in generate-start log (was a NameError risk)
        m = re.search(
            r"arc_eval_log\(\s*\n\s*f\"\[ARC-PHASE-1\] Generate start:.*?\)",
            SOURCE,
            re.DOTALL,
        )
        self.assertIsNotNone(m, "Generate start log block missing")
        self.assertNotIn("slot_temp", m.group(0))

    def test_ast_parses(self):
        ast.parse(SOURCE)


class TestTokenBudgetMath(unittest.TestCase):
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

    def _arc_hypothesis_max_tokens(self, task):
        budget = int(self.ARC_MBR_OUTPUT_TOKEN_BUDGET)
        final_reserve = max(
            self._arc_final_grid_max_tokens(task),
            int(budget * self.ARC_FINAL_GRID_MIN_FRACTION),
        )
        hyp_pool = max(0, budget - final_reserve)
        return max(64, hyp_pool // max(1, self.ARC_HYPOTHESIS_SLOTS))

    def test_per_slot_budget_is_875_not_7000(self):
        task = {
            "train": [
                {"input": [[0]], "output": [[0] * 30 for _ in range(30)]},
            ],
            "test": [{"input": [[0] * 30 for _ in range(30)]}],
        }
        per_slot = self._arc_hypothesis_max_tokens(task)
        self.assertEqual(per_slot, 875)
        self.assertLess(per_slot, 2000, "per-slot cap should not look like full 7k budget")


class TestCollectFeatureSlotHypothesesMock(unittest.TestCase):
    def _import_module_quiet(self):
        import io
        import importlib.util
        import types

        # Windows / CPU envs often lack vllm._C — stub vllm before loading the script.
        class _FakeSamplingParams:
            def __init__(self, **kwargs):
                for key, val in kwargs.items():
                    setattr(self, key, val)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = MagicMock
        fake_vllm.SamplingParams = _FakeSamplingParams
        fake_sp_mod = types.ModuleType("vllm.sampling_params")
        fake_sp_mod.SamplingParams = _FakeSamplingParams

        buf = io.StringIO()
        spec = importlib.util.spec_from_file_location("mbr_test", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "vllm": fake_vllm,
                "vllm.sampling_params": fake_sp_mod,
            },
        ), patch("sys.stdout", buf), patch("sys.stderr", buf):
            spec.loader.exec_module(mod)
        return mod

    def test_collect_calls_generate_with_use_tqdm_false(self):
        try:
            mbr = self._import_module_quiet()
        except Exception as exc:
            self.skipTest(f"full module import unavailable in this env: {exc}")

        captured = {}

        class FakeOut:
            def __init__(self, n: int):
                self.outputs = [MagicMock(token_ids=list(range(10)))]

        def fake_generate(prompts, sp_list, use_tqdm=True):
            captured["use_tqdm"] = use_tqdm
            captured["n_prompts"] = len(prompts)
            captured["max_tokens"] = [sp.max_tokens for sp in sp_list]
            captured["temps"] = [sp.temperature for sp in sp_list]
            return [FakeOut(i) for i in range(len(prompts))]

        llm = MagicMock()
        llm.generate = fake_generate
        llm.max_model_len = 16384
        llm.llm_engine = None

        tok = MagicMock()
        tok.decode = lambda ids, skip_special_tokens=True: "TRANSFORMATION_HYPOTHESIS: rotate 90"
        tok.pad_token = "<pad>"
        tok.apply_chat_template = lambda messages, tokenize=False, add_generation_prompt=True: "chat"

        task = {
            "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
            "test": [{"input": [[1, 0], [0, 1]]}],
        }

        fake_alloc_out = {
            "feature_indices": list(range(8)),
            "feature_names": [f"Feat{i}" for i in range(8)],
        }
        fake_allocator = MagicMock()
        fake_allocator.return_value = fake_alloc_out
        fake_allocator.get_feature_params.return_value = [
            {"temperature": 0.7, "top_p": 0.92, "repetition_penalty": 1.03}
        ] * 8

        with patch.object(mbr, "count_prompt_tokens", return_value=100), patch.object(
            mbr, "make_pooled_state", return_value=MagicMock()
        ), patch.object(
            mbr, "PermutationFeatureSlotAllocator", return_value=fake_allocator
        ):
            hyps = mbr.collect_feature_slot_hypotheses(
                llm,
                tok,
                "test_task",
                task,
                test_index=0,
                k=8,
                verbose=False,
            )

        self.assertFalse(captured.get("use_tqdm", True))
        self.assertEqual(captured.get("n_prompts"), 8)
        self.assertEqual(len(hyps), 8)
        self.assertTrue(all(t == 0.0 for t in captured.get("temps", [])))
        self.assertTrue(all(mt == 875 for mt in captured.get("max_tokens", [])))


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSourceStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenBudgetMath))
    suite.addTests(loader.loadTestsFromTestCase(TestCollectFeatureSlotHypothesesMock))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n=== SUMMARY ===")
    print(f"tests={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())