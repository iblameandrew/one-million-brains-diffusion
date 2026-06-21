#!/usr/bin/env python3
"""Remove multi-agent voter pool from million_brains_dflash.py."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "million_brains_dflash.py"

REMOVE_NAMES = {
    "_build_vllm_worker_fast_attempts",
    "_worker_emit_progress",
    "_collect_script_source_candidates",
    "_is_materialized_worker_library_path",
    "_worker_library_markers",
    "_script_text_has_worker_markers",
    "_script_source_sort_key",
    "_get_notebook_cell_source_bytes",
    "_find_running_script_source",
    "_validate_worker_library_file",
    "_canonical_worker_script_path",
    "_canonical_voter_worker_entry_path",
    "_find_voter_worker_entry_source",
    "_materialize_bytes_to_path",
    "_resolve_live_worker_library_bytes",
    "_worker_library_target_path",
    "_worker_library_needs_refresh",
    "_materialize_worker_script_path",
    "_materialize_voter_worker_entry_path",
    "_voter_pool_worker_cwd",
    "_resolve_multi_agent_worker_script_path",
    "_resolve_voter_pool_gen_path",
    "_read_worker_stream_line",
    "_read_worker_json_message",
    "_multi_agent_gpu_util_per_engine",
    "_voter_pool_agent_waves",
    "_voter_worker_subprocess_env",
    "_peek_worker_stderr_tail",
    "_canonical_grid_vote_key",
    "plurality_vote_agent_grids",
    "_multi_agent_worker_execute_mbr",
    "_WorkerStdoutToStderr",
    "_worker_redirect_stdout_fd_to_stderr",
    "_worker_restore_stdout_fd",
    "_multi_agent_worker_main",
    "MultiAgentEnginePool",
    "create_multi_agent_pool",
    "print_multi_agent_vote_summary",
}

REPLACEMENTS = [
    (
        'SCRIPT_VERSION = "2026-06-19-diffusion-c"  # removed speculative/DFlash legacy',
        'SCRIPT_VERSION = "2026-06-19-diffusion-d"  # single-engine ARC (no voter pool)',
    ),
    (
        "# Multi-agent layer: N independent DiffusionGemma workers (1 per GPU) + plurality vote on final grid.\n"
        "ARC_MULTI_AGENT_ENABLED = True  # plurality vote across N independent voter workers\n"
        "ARC_MULTI_AGENT_REQUIRED = True  # ARC eval: never silently fall back to single engine\n"
        "ARC_MULTI_AGENT_N = 4  # 1/GPU on 4x L4 (5+ shares VRAM → KV OOM); raise only with 18i worker caps\n"
        "ARC_MULTI_AGENT_GPU_UTIL = 0.0  # 0 = auto (~0.88 / ceil(N / num_gpus)) per engine\n"
        "ARC_WORKER_VLLM_MAX_MODEL_LEN = 10240  # 30x30 terse prompts need ~9600 tok (8192 too small)\n"
        "ARC_WORKER_VLLM_GPU_UTIL_CAP = 0.70  # headroom for KV at 8k ctx (0.88 OOMs)\n"
        "ARC_PHASE1_PROMPT_PARALLELISM = False  # False = 1 hypothesis slot per vLLM generate() (reliable on L4)\n"
        "ARC_WORKER_VLLM_MAX_NUM_SEQS = 1  # match sequential Phase-1 (no multi-prompt batching)\n"
        "ARC_WORKER_VLLM_SHARED_MAX_MODEL_LEN = 4096  # wave-1+ engines sharing a GPU with wave-0\n"
        "ARC_WORKER_VLLM_FAST_LOAD = True  # voter subprocesses: 1-2 vLLM attempts, enforce_eager=True\n"
        "ARC_WORKER_LOAD_GPU_WAVES = True  # one vLLM load per GPU at a time (wave 0..N)\n"
        "ARC_VOTER_INFER_GPU_WAVES = True  # inference: one active voter per GPU at a time\n"
        "ARC_SPATIAL_SEQUENTIAL_SLOTS_WHEN_SHARED_GPU = True  # when parallelism on: 1 slot/gen if 2 engines/GPU\n"
        'ARC_MULTI_AGENT_VOTE = "plurality"  # plurality = most common parsed grid; ties -> lowest agent id\n'
        "MBR_WORKER_SCRIPT_PATH = None  # None = auto; set e.g. /kaggle/working/million_brains_dflash.py in notebooks\n",
        "ARC_PHASE1_PROMPT_PARALLELISM = False  # False = 1 hypothesis slot per vLLM generate()\n"
        "ARC_VLLM_MAX_NUM_SEQS = 1  # max Phase-1 slots batched per vLLM call when parallelism on\n",
    ),
]

PRINT_POST_LOAD = '''def print_post_load_arc_config() -> None:
    """Short cheat-sheet printed right after model load (before ARC eval banner)."""
    hyp_n = arc_hypothesis_k()
    print("\\n" + "-" * 72)
    print("RUNTIME ARC CONFIG — QUICK REFERENCE")
    print("-" * 72)
    print("  Engine       : single DiffusionGemma vLLM")
    print(f"  Per test     : Phase1:{hyp_n} props -> Phase2:1 grid (pixel vote if spatial ensemble)")
    print(f"  BENCHMARK_K  : {K} (demo benchmark only — NOT used in ARC Phase 1/2)")
    print(
        f"  vLLM hints   : Phase1 shows 'Rendering prompts: {hyp_n}/{hyp_n}' | "
        f"Phase2 shows 'Rendering prompts: 1/1'"
    )
    print("-" * 72 + "\\n")

'''

PRINT_ARC_PIPELINE = '''def print_arc_pipeline_architecture(
    *,
    vllm_llm: Optional[Any] = None,
) -> None:
    """One-screen map of ARC eval logging."""
    hyp_n = arc_hypothesis_k()
    phase1_mode = (
        "1 slot/vLLM call (parallelism off)"
        if not ARC_PHASE1_PROMPT_PARALLELISM
        else f"up to {ARC_VLLM_MAX_NUM_SEQS} slots/vLLM call"
    )
    print("\\n" + "=" * 80)
    print("ARC PIPELINE — HOW TO READ THE LOGS")
    print("=" * 80)
    if ARC_SPATIAL_GRID_ENSEMBLE:
        phase1_line = (
            f"    [ARC-PHASE-1]  Spatial pool     -> {hyp_n} JSON grid hypotheses "
            f"(guided={ARC_GUIDED_JSON_DECODING}, thinking=False, {phase1_mode})\\n"
        )
        phase2_line = (
            "    [ARC-PHASE-2]  Pixel majority   -> per-cell vote across parsed grids\\n"
        )
    else:
        phase1_line = (
            f"    [ARC-PHASE-1]  Hypothesis pool  -> {hyp_n} parallel TEXT proposals\\n"
        )
        phase2_line = (
            "    [ARC-PHASE-2]  Final grid       -> 1 JSON grid synthesis\\n"
            "                   vLLM shows: Rendering prompts: 1/1\\n"
        )
    print(
        "  Single engine runs TWO phases per test case:\\n"
        f"{phase1_line}"
        f"                   vLLM shows: Rendering prompts: {hyp_n}/{hyp_n}  (= hypothesis slot count)\\n"
        f"{phase2_line}"
        "  Note: Phase-1 can look idle at 'Processed 0/N' until the first slot fully completes."
    )
    print("")
    print(
        f"  BENCHMARK_K={K} is for the demo/benchmark path only — "
        f"ARC hypothesis count is ARC_HYPOTHESIS_SLOTS={ARC_HYPOTHESIS_SLOTS}"
    )
    print("=" * 80 + "\\n")

'''


def delete_ranges(lines: list[str], ranges: list[tuple[int, int]]) -> list[str]:
    drop = set()
    for start, end in ranges:
        for i in range(start, end + 1):
            drop.add(i)
    return [line for i, line in enumerate(lines, start=1) if i not in drop]


def replace_function_block(text: str, name: str, replacement: str) -> str:
    pattern = rf"(?ms)^def {re.escape(name)}\(.*?\n(?=^def |^class |^@dataclass|^# ={{10,}}|^if __name__)"
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"function not found: {name}")
    return text[: m.start()] + replacement + text[m.end() :]


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        if old not in text:
            raise RuntimeError(f"missing anchor: {old[:60]}...")
        text = text.replace(old, new, 1)

    tree = ast.parse(text)
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in REMOVE_NAMES:
            ranges.append((node.lineno, node.end_lineno))
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_EMBEDDED_VOTER_WORKER_ENTRY":
                    ranges.append((node.lineno, node.end_lineno))

    text = "".join(delete_ranges(text.splitlines(keepends=True), sorted(ranges)))

    text = replace_function_block(text, "print_post_load_arc_config", PRINT_POST_LOAD)
    text = replace_function_block(text, "print_arc_pipeline_architecture", PRINT_ARC_PIPELINE)

    # arc_phase1_slot_batch_size
    text = re.sub(
        r"def arc_phase1_slot_batch_size\(k: int\) -> int:.*?return min\(max\(1, int\(k\)\), max\(1, int\(ARC_WORKER_VLLM_MAX_NUM_SEQS\)\)\)",
        '''def arc_phase1_slot_batch_size(k: int) -> int:
    """How many Phase-1 slots to pass per vLLM generate() call."""
    if not ARC_PHASE1_PROMPT_PARALLELISM:
        return 1
    return min(max(1, int(k)), max(1, int(ARC_VLLM_MAX_NUM_SEQS)))''',
        text,
        count=1,
        flags=re.S,
    )

    text = text.replace("_worker_emit_progress(", "arc_eval_log(")
    text = text.replace(" or _MBR_WORKER_SUBPROCESS", "")
    text = re.sub(
        r"\n    attempts: List\[Dict\[str, Any\]\] = \[kwargs\]\n"
        r"    if _MBR_WORKER_SUBPROCESS and ARC_WORKER_VLLM_FAST_LOAD:.*?"
        r"        \)\n",
        "\n    attempts: List[Dict[str, Any]] = [kwargs]\n",
        text,
        count=1,
        flags=re.S,
    )

    # evaluate_arc_dataset signature + multi-agent branches
    text = re.sub(
        r"    visual_grading: bool = ARC_VISUAL_GRADING,\n"
        r"    multi_agent_pool: Optional\[MultiAgentEnginePool\] = None,\n"
        r"    voter_pool_single_reason: Optional\[str\] = None,\n",
        "    visual_grading: bool = ARC_VISUAL_GRADING,\n",
        text,
    )
    text = re.sub(
        r"    print_arc_pipeline_architecture\(\n"
        r"        multi_agent_pool=multi_agent_pool,\n"
        r"        vllm_llm=vllm_llm,\n"
        r"    \)\n"
        r"    if multi_agent_pool is not None:.*?print\(\"-\" \* 80\)\n\n",
        "    print_arc_pipeline_architecture(vllm_llm=vllm_llm)\n"
        '    print(f"[CONFIG] Single engine | Phase1={arc_hypothesis_k()} props/test | Phase2=1 grid/test")\n'
        "    if visual_grading and ARC_SAVE_GRADE_IMAGES:\n"
        '        print(f"Grade images: {_arc_grade_output_dir()}/")\n'
        '    print("-" * 80)\n\n',
        text,
        count=1,
        flags=re.S,
    )

    # task loop: remove multi-agent preamble, keep single path only
    text = re.sub(
        r"            hyp_n = arc_hypothesis_k\(\)\n"
        r"            if multi_agent_pool is not None:.*?"
        r"                \)\n"
        r"            t0 = time\.perf_counter\(\)\n"
        r"            agent_results: List\[Dict\[str, Any\]\] = \[\]\n"
        r"            vote_meta: Dict\[str, Any\] = \{\}\n"
        r"            mbr_res: Dict\[str, Any\] = \{\}\n"
        r"            if multi_agent_pool is not None:.*?"
        r"                \)\n"
        r"            elif ARC_SLOT_HYPOTHESIS_MODE:",
        '''            hyp_n = arc_hypothesis_k()
            phase2 = (
                "pixel majority vote"
                if ARC_SPATIAL_GRID_ENSEMBLE
                else "Phase2:1 grid"
            )
            arc_eval_log(
                f"\\n[ARC] >>> task {task_idx + 1}/{len(task_ids)} {task_id} "
                f"test#{test_index} — Phase1:{hyp_n} "
                f"{'spatial grids' if ARC_SPATIAL_GRID_ENSEMBLE else 'props'} -> {phase2}"
            )
            t0 = time.perf_counter()
            mbr_res: Dict[str, Any] = {}
            if ARC_SLOT_HYPOTHESIS_MODE:''',
        text,
        count=1,
        flags=re.S,
    )

    text = re.sub(
        r"            if multi_agent_pool is None:\n"
        r"                arc_eval_log\(\n"
        r'                    f"\[ARC\] <<< task.*?'
        r"                \)\n"
        r"            else:\n"
        r"                arc_eval_log\(\n"
        r'                    f"\[ARC\] <<< task.*?'
        r"                \)\n",
        '''            arc_eval_log(
                f"[ARC] <<< task {task_idx + 1}/{len(task_ids)} {task_id} "
                f"test#{test_index} — done | {format_timing_line(mbr_timing)} | "
                f"phase1_hyp_tok={mbr_res.get('hypothesis_tokens', 0)} "
                f"phase2_grid_tok={mbr_res.get('grid_tokens', 0)} "
                f"budget={mbr_res.get('output_budget_cap', ARC_MBR_OUTPUT_TOKEN_BUDGET)}"
            )
''',
        text,
        count=1,
        flags=re.S,
    )

    text = text.replace(
        '"multi_agent_vote": vote_meta if multi_agent_pool is not None else None,',
        '"multi_agent_vote": None,',
    )

    # main entry
    text = re.sub(
        r"    if \"--mbr-agent-worker\" in sys\.argv:.*?"
        r"        raise SystemExit\(0\)\n\n",
        "",
        text,
        count=1,
        flags=re.S,
    )

    text = re.sub(
        r"    print\(\n"
        r"        f\"    ARC_MULTI_AGENT_ENABLED=\{ARC_MULTI_AGENT_ENABLED\}.*?\"\n"
        r"    \)\n",
        "",
        text,
        count=1,
        flags=re.S,
    )

    text = re.sub(
        r"    # 3\) Choose & load model \(voter pool is default for ARC eval; single engine for demo only\)\n"
        r"    model_name = \"unknown\"\n"
        r"    multi_agent_pool: Optional\[MultiAgentEnginePool\] = None\n"
        r"    voter_pool_single_reason: Optional\[str\] = None\n"
        r"    vllm_llm: Optional\[Any\] = None\n"
        r"    tokenizer: Any = None\n"
        r"    hf_model: Optional\[Any\] = None\n"
        r"    run_arc_eval = .*?\n"
        r"    use_multi_agent = ARC_MULTI_AGENT_ENABLED and run_arc_eval\n"
        r"    print\(.*?\n"
        r"    \)\n"
        r"    if use_multi_agent:.*?elif PREFER_LOCAL_MODELS:\n"
        r"        try:\n"
        r"            vllm_llm, tokenizer, hf_model = load_local_models\(\)\[:3\]\n"
        r"            model_name = resolve_local_model_path\(\) or KAGGLE_DIFFUSIONGEMMA_DIR\n"
        r"        except RuntimeError as _local_e:\n"
        r"            print\(_local_e\)\n"
        r"            print\(\"\[LOCAL-LOAD\] Falling back to remote model resolution\.\.\.\"\)\n"
        r"            model_name, _backend = pick_model_name\(\)\n"
        r"            vllm_llm, tokenizer, hf_model = load_models\(model_name\)\n"
        r"    else:\n"
        r"        model_name, _backend = pick_model_name\(\)\n"
        r"        vllm_llm, tokenizer, hf_model = load_models\(model_name\)\n",
        '''    model_name = "unknown"
    vllm_llm: Optional[Any] = None
    tokenizer: Any = None
    hf_model: Optional[Any] = None
    run_arc_eval = (
        not args.demo_only
        and args.eval_challenges
        and args.eval_solutions
    )
    if PREFER_LOCAL_MODELS:
        try:
            vllm_llm, tokenizer, hf_model = load_local_models()[:3]
            model_name = resolve_local_model_path() or KAGGLE_DIFFUSIONGEMMA_DIR
        except RuntimeError as _local_e:
            print(_local_e)
            print("[LOCAL-LOAD] Falling back to remote model resolution...")
            model_name, _backend = pick_model_name()
            vllm_llm, tokenizer, hf_model = load_models(model_name)
    else:
        model_name, _backend = pick_model_name()
        vllm_llm, tokenizer, hf_model = load_models(model_name)
''',
        text,
        count=1,
        flags=re.S,
    )

    text = re.sub(
        r"    if multi_agent_pool is None:\n"
        r"        verify_inference_engine\(vllm_llm, tokenizer\)\n"
        r"    else:\n"
        r"        print\(.*?\n"
        r"        \)\n\n"
        r"    if run_arc_eval:\n"
        r"        print_post_load_arc_config\(\n"
        r"            multi_agent_pool,\n"
        r"            single_engine_reason=voter_pool_single_reason,\n"
        r"        \)\n",
        '''    verify_inference_engine(vllm_llm, tokenizer)

    if run_arc_eval:
        print_post_load_arc_config()
''',
        text,
        count=1,
        flags=re.S,
    )

    text = re.sub(
        r"            visual_grading=ARC_VISUAL_GRADING and not args\.no_arc_visuals,\n"
        r"            multi_agent_pool=multi_agent_pool,\n"
        r"            voter_pool_single_reason=voter_pool_single_reason,\n",
        "            visual_grading=ARC_VISUAL_GRADING and not args.no_arc_visuals,\n",
        text,
    )

    text = re.sub(
        r"    if run_demo:\n"
        r"        if vllm_llm is None:\n"
        r"            print\(\n"
        r'                "\[demo\] Skipped — voter-pool mode has no parent vLLM engine .*?"\n'
        r"            \)\n"
        r"        else:\n"
        r"            results\[\"demo\"\] = benchmark\(\n"
        r"                vllm_llm, tokenizer, BENCHMARK_PROMPT, max_new=TARGET_MAX_TOKENS\n"
        r"            \)\n\n"
        r"    if multi_agent_pool is not None:\n"
        r"        multi_agent_pool\.shutdown\(\)\n",
        '''    if run_demo:
        results["demo"] = benchmark(
            vllm_llm, tokenizer, BENCHMARK_PROMPT, max_new=TARGET_MAX_TOKENS
        )

''',
        text,
        count=1,
        flags=re.S,
    )

    text = re.sub(
        r'            "arc_multi_agent_n": ARC_MULTI_AGENT_N,\n'
        r'            "voter_pool": \(\n'
        r"                f\"multi_x\{multi_agent_pool\.n_agents\}\"\n"
        r"                if multi_agent_pool is not None\n"
        r'                else "single"\n'
        r"            \),\n",
        "",
        text,
    )

    text = text.replace("or raise ARC_WORKER_VLLM_MAX_MODEL_LEN.", "or raise DIFFUSION_MAX_MODEL_LEN.")
    text = re.sub(
        r"# ={10,}\n# ONE-MILLION-BRAINS GENERATE \(compat wrapper \+ legacy autoregressive path\)\n# ={10,}\n",
        "",
        text,
        count=1,
    )

    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {TARGET} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()