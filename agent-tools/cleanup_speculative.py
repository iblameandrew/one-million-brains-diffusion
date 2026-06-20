#!/usr/bin/env python3
"""Remove speculative-decoding / DFlash legacy code from million_brains_dflash.py."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "million_brains_dflash.py"

REMOVE_FUNCS = {
    "_resolve_vllm_draft_module",
    "_million_brains_autoregressive_generate",
    "_live_edit_dflash",
    "is_dflash_draft_checkpoint",
    "discover_qwen_dflash_bundle_paths",
    "find_paired_dflash_draft",
    "find_paired_base_checkpoint",
    "_vllm_native_speculative_viable",
    "_vllm_spec_attempt_should_abort",
    "_vllm_supported_speculative_methods",
    "_resolve_vllm_speculative_method",
    "_vllm_dflash_draft_speculative_viable",
    "_vllm_speculative_load_viable",
    "_build_vllm_speculative_config",
    "_inference_speculative_status",
}

REMOVE_CLASSES = {"MillionBrainsDFlashDraftModel"}

# Inclusive line ranges to delete (module-level blocks not always top-level AST)
EXTRA_RANGES = [
    (457, 462),  # ENABLE_DFLASH_LIVE_PATCH vllm_draft_module stub
    (2163, 2196),  # MillionBrainsDFlashDraftModel + section header before _live_edit
    (2373, 2390),  # live-edit startup if/else
]

NEW_HEADER = '''#!/usr/bin/env python3
"""
million_brains_dflash.py — ONE-MILLION-BRAINS-DIFFUSIONGEMMA
Permutation-Gated Feature-Slot Allocator hard-wired into DiffusionGemma canvas denoising.

Architecture: DiffusionGemma block-diffusion (256-token canvas, iterative denoising) with
K parallel Million-Brains conditioned trajectories per denoise step — feature-slot allocation,
CTSB smoothing, cross-stream integration, cumprod verification, and adaptive reallocation.

Kaggle: attach Models input google/diffusiongemma, run as script or notebook cell.
"""

'''

REPLACEMENTS: list[tuple[str, str]] = [
    (
        'SCRIPT_VERSION = "2026-06-19-diffusion-b"  # Kaggle Models path for DiffusionGemma',
        'SCRIPT_VERSION = "2026-06-19-diffusion-c"  # removed speculative/DFlash legacy',
    ),
    (
        "USE_DIFFUSIONGEMMA = True  # True = DiffusionGemma denoising + MBR conditioning (no DFlash draft)\n",
        "",
    ),
    (
        "# --- DFlash / speculative decoding: PERMANENTLY DISABLED (DiffusionGemma replaces draft path) ---\n"
        "ENABLE_DFLASH_LIVE_PATCH = False  # never enable — DFlash monkey-patches removed from hot path\n"
        "ENABLE_VLLM_SPECULATIVE_DECODING = False  # never enable — no draft_model / DFlash spec\n",
        "",
    ),
    (
        "BLOCK_SIZE = DIFFUSION_DENOISE_CHUNK if USE_DIFFUSIONGEMMA else 6  # tokens per denoise step / super-block",
        "BLOCK_SIZE = DIFFUSION_DENOISE_CHUNK  # tokens committed per denoise step / super-block",
    ),
    (
        "# Kaggle notebook input root (Add Input -> Notebooks -> godelcomplete/qwen3-5-4b-dflash)\n"
        'KAGGLE_QWEN_BUNDLE_ROOT = "/kaggle/input/notebooks/godelcomplete/qwen3-5-4b-dflash"\n'
        "LOCAL_MODEL_PATH = KAGGLE_QWEN_BUNDLE_ROOT\n"
        'LOCAL_BASE_DIR = f"{KAGGLE_QWEN_BUNDLE_ROOT}/Qwen3.5-4B"  # BASE Qwen — generation target (NOT draft)\n'
        'LOCAL_DFLASH_DIR = f"{KAGGLE_QWEN_BUNDLE_ROOT}/Qwen3.5-4B-DFlash"  # DFlash draft only\n'
        'BASE_BUNDLE_DIR_NAMES = frozenset({"qwen3.5-4b", "qwen3-5-4b"})\n'
        'DRAFT_BUNDLE_DIR_NAMES = frozenset({"qwen3.5-4b-dflash", "qwen3-5-4b-dflash"})\n'
        "# vLLM load safety for custom DFlash / Qwen3.5 checkpoints (avoid pooling + torch.compile crash)\n",
        "# Local model path override (optional; DiffusionGemma resolved via KAGGLE_DIFFUSIONGEMMA_DIR)\n"
        "LOCAL_MODEL_PATH = KAGGLE_DIFFUSIONGEMMA_DIR\n",
    ),
    (
        "VLLM_GPU_MEMORY_UTILIZATION = 0.88  # base-only fallback; spec uses VLLM_SPEC_GPU_MEMORY_UTILIZATION\n",
        "VLLM_GPU_MEMORY_UTILIZATION = 0.88  # HF fallback only; DiffusionGemma uses DIFFUSION_GPU_UTIL\n",
    ),
    (
        "# Legacy speculative knobs (inactive when USE_DIFFUSIONGEMMA=True; kept for API compat only)\n"
        "VLLM_REQUIRE_SPECULATIVE = False  # must stay False on Kaggle — spec attempts loop-crash the kernel\n"
        "VLLM_SINGLE_ENGINE_SPECULATIVE = False  # only skips multi-agent when native vLLM spec is ON\n"
        "VLLM_LANGUAGE_MODEL_ONLY = True  # Qwen3.5 text-only: skip vision encoder; required for draft_model spec on stock vLLM\n"
        'VLLM_SPECULATIVE_METHOD = "auto"  # auto: dflash if vLLM supports it, else draft_model (stock Kaggle vLLM)\n'
        "VLLM_NUM_SPECULATIVE_TOKENS = BLOCK_SIZE  # DFlash block width per speculation step\n"
        "VLLM_SPEC_GPU_MEMORY_UTILIZATION = 0.90  # draft+target needs ~90% of 22GB L4 (0.70 left no room for draft)\n"
        "VLLM_SPEC_DRAFT_GPU_UTIL_SCALE = 1.0  # do not shrink util for target+draft\n"
        "VLLM_SPEC_MAX_MODEL_LEN = 16384  # cap KV for spec load (prompt resolver shrinks; 22272 OOMs with draft on L4)\n"
        "VLLM_SPEC_PREFER_TENSOR_PARALLEL = 1  # tp=1 first (tp=2 spawns NCCL workers that spam logs on Kaggle)\n"
        "VLLM_SPEC_TRY_SMALLEST_CONTEXT_FIRST = True  # 8192→16384 ascending; fits draft+target before huge KV reserve\n",
        "",
    ),
    (
        "EVAL_MAX_NEW_TOKENS = 512  # per-task budget for token-speculative ARC path",
        "EVAL_MAX_NEW_TOKENS = 512  # per-task generation budget for ARC eval",
    ),
    (
        "ARC_MULTI_AGENT_DISABLE_SPECULATIVE = True  # base-only per agent — spec+draft OOMs at 2 engines/GPU\n",
        "",
    ),
    (
        "# CUMPROD ACCEPTANCE (core speculative primitive, generalized to K paths)",
        "# CUMPROD ACCEPTANCE (MBR verification primitive for K parallel trajectories)",
    ),
    (
        "    Walk the proposed block left-to-right performing the classic speculative test.",
        "    Walk the proposed block left-to-right performing cumprod acceptance vs target logprobs.",
    ),
    (
        "            # In full speculative we would draw a new token from (target - draft) here.",
        "            # Resample-from-adjusted-distribution step omitted in this orchestration layer.",
    ),
    (
        "    In a real kernel patch (inside DFlashDraftModel or a custom attention forward)",
        "    In a future kernel integration (custom attention forward inside the denoiser)",
    ),
]

RESOLVE_GENERATION_MODEL_PATH = '''def resolve_generation_model_path(requested_path: str) -> str:
    """Resolve DiffusionGemma checkpoint path."""
    resolved = resolve_checkpoint_dir(requested_path) or requested_path
    if not local_dir_exists(resolved):
        resolved = resolve_diffusiongemma_model_path()
    print(f"[LOAD] DiffusionGemma checkpoint: {resolved}")
    return resolved

'''

MILLION_BRAINS_GENERATE = '''# =============================================================================
# ONE-MILLION-BRAINS GENERATE (public API — DiffusionGemma conditioned denoising)
# =============================================================================
def million_brains_dflash_generate(
    vllm_llm: LLM,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    k: int = 4,
    block_size: int = 6,
    allocator: Optional[PermutationFeatureSlotAllocator] = None,
    enable_reallocation: bool = True,
    enable_smoothing: bool = ENABLE_CIRCUIT_SMOOTHING,
    seed: int = 42,
    verbose: bool = STREAM_ALL_OUTPUT,
    arc_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Million-Brains conditioned DiffusionGemma denoising (compat entry point)."""
    return million_brains_diffusion_denoise_generate(
        vllm_llm,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        k=k,
        block_size=block_size,
        allocator=allocator,
        enable_reallocation=enable_reallocation,
        enable_smoothing=enable_smoothing,
        seed=seed,
        verbose=verbose,
        arc_context=arc_context,
    )


'''

BANNER_BLOCK = '''# =============================================================================
# REQUIRED BANNER
# =============================================================================
def print_one_million_brains_banner(success: bool = True):
    banner = r"""
================================================================================
 ██████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗██╗     ██╗     ██╗ ██████╗ ███╗   ██╗    ██████╗ ██████╗  █████╗ ██╗███╗   ██╗███████╗    ███████╗██╗      █████╗ ███████╗██╗  ██╗
██╔═══██╗████╗  ██║██╔════╝    ████╗ ████║██║██║     ██║     ██║██╔═══██╗████╗  ██║    ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝    ██╔════╝██║     ██╔══██╗██╔════╝██║  ██║
██║   ██║██╔██╗ ██║█████╗      ██╔████╔██║██║██║     ██║     ██║██║   ██║██╔██╗ ██║    ██████╔╝██████╔╝███████║██║██╔██╗ ██║███████╗    █████╗  ██║     ███████║███████╗███████║
██║   ██║██║╚██╗██║██╔══╝      ██║╚██╔╝██║██║██║     ██║     ██║██║   ██║██║╚██╗██║    ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║╚════██║    ██╔══╝  ██║     ██╔══██║╚════██║██╔══██║
╚██████╔╝██║ ╚████║███████╗    ██║ ╚═╝ ██║██║███████╗███████╗██║╚██████╔╝██║ ╚████║    ██████╔╝██║  ██║██║  ██║██║██║ ╚████║███████║    ██║     ███████╗██║  ██║███████║██║  ██║
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
================================================================================
"""
    print(banner)
    if success:
        print(
            " ONE-MILLION-BRAINS-DIFFUSIONGEMMA INITIALIZED  |  "
            "CANVAS=%d  |  DENOISE_K=%d  |  FEATURES=%d  |  REALLOCATION=%s"
            % (
                DIFFUSION_CANVAS_LENGTH,
                K,
                NUM_PERSONALITY_FEATURES,
                str(ENABLE_FEATURE_REALLOCATION).upper(),
            )
        )
        print(
            " Engine: DiffusionGemma block-diffusion + hard-wired MBR conditioned denoising"
        )
        print(f" Script version: {SCRIPT_VERSION}")
    else:
        print(
            " ONE-MILLION-BRAINS-DIFFUSIONGEMMA INITIALIZED (DEGRADED — load errors, HF fallback may be active)"
        )
    print(
        "================================================================================\\n"
    )


if not _MBR_WORKER_SUBPROCESS:
    print(
        "[CONFIG] DiffusionGemma + Million-Brains conditioned denoising "
        f"(model={DIFFUSIONGEMMA_MODEL_PRIMARY})",
        flush=True,
    )
    print_one_million_brains_banner(True)


'''

ADAPTIVE_GPU_UTIL = '''def _adaptive_vllm_gpu_util(requested: float) -> float:
    """Cap gpu_memory_utilization so vLLM's requested slice fits in free VRAM."""
    snap = _cuda_vram_snapshot()
    if snap["total_gib"] <= 0:
        return requested
    safe = (snap["free_gib"] / snap["total_gib"]) * 0.88
    util = min(float(requested), safe)
    util = max(0.38, min(0.88, util))
    print(
        f"[LOAD] VRAM cuda:0 "
        f"free {snap['free_gib']:.2f}/{snap['total_gib']:.2f} GiB "
        f"(allocated {snap['allocated_gib']:.2f}) "
        f"-> gpu_memory_utilization={util:.2f}"
    )
    return util

'''

BUILD_VLLM_ATTEMPTS = '''def _build_vllm_attempts(
    model_path: str,
    gpu_memory_utilization: float,
) -> List[Dict[str, Any]]:
    kwargs = _build_diffusiongemma_vllm_kwargs(
        model_path, gpu_memory_utilization=gpu_memory_utilization
    )
    return [kwargs]

'''

BUILD_WORKER_ATTEMPTS = '''def _build_vllm_worker_fast_attempts(
    model_path: str,
    gpu_memory_utilization: float,
) -> List[Dict[str, Any]]:
    """Minimal vLLM load grid for voter subprocesses."""
    return [
        _build_diffusiongemma_vllm_kwargs(
            model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=ARC_WORKER_VLLM_MAX_MODEL_LEN or DIFFUSION_MAX_MODEL_LEN,
            max_num_seqs=1,
        )
    ]

'''

CREATE_INFERENCE_ENGINE = '''def create_inference_engine(
    model_path: str,
    tokenizer: Any,
    *,
    gpu_memory_utilization: Optional[float] = None,
) -> Any:
    """Create DiffusionGemma vLLM engine (HF fallback on load failure)."""
    gen_path = resolve_generation_model_path(model_path)
    local_only = os.path.isdir(gen_path)
    gpu_mem = (
        float(gpu_memory_utilization)
        if gpu_memory_utilization is not None
        else float(DIFFUSION_GPU_UTIL)
    )

    kwargs = _build_diffusiongemma_vllm_kwargs(gen_path, gpu_memory_utilization=gpu_mem)
    print(
        f"[LOAD] DiffusionGemma vLLM: canvas={DIFFUSION_CANVAS_LENGTH} "
        f"max_seqs={kwargs.get('max_num_seqs')} "
        f"max_len={kwargs.get('max_model_len')} "
        f"gpu_util={kwargs.get('gpu_memory_utilization')}",
        flush=True,
    )
    attempts: List[Dict[str, Any]] = [kwargs]
    if _MBR_WORKER_SUBPROCESS and ARC_WORKER_VLLM_FAST_LOAD:
        attempts = _build_vllm_worker_fast_attempts(gen_path, gpu_mem)
        print(
            f"[LOAD] Worker fast vLLM load: {len(attempts)} attempt(s), "
            f"max_model_len={attempts[0].get('max_model_len')}",
            flush=True,
        )

    last_err: Optional[BaseException] = None
    _release_cuda_cache()
    for attempt_idx, attempt_kwargs in enumerate(attempts, start=1):
        try:
            print(
                f"[LOAD] vLLM attempt {attempt_idx}/{len(attempts)}: "
                f"max_model_len={attempt_kwargs.get('max_model_len')} "
                f"gpu_util={attempt_kwargs.get('gpu_memory_utilization')}"
            )
            llm = LLM(**attempt_kwargs)
            ctx = int(attempt_kwargs.get("max_model_len", DIFFUSION_MAX_MODEL_LEN))
            setattr(llm, "max_model_len", ctx)
            setattr(llm, "diffusiongemma_enabled", True)
            print(f"[LOAD] DiffusionGemma engine ready (max_model_len={ctx})")
            return llm
        except Exception as exc:
            last_err = exc
            print(f"[LOAD] vLLM attempt {attempt_idx} failed: {type(exc).__name__}: {exc}")
            _release_cuda_cache()

    if VLLM_FALLBACK_TO_HF:
        print("[LOAD] All vLLM attempts failed; using HuggingFace generate fallback.")
        return HFGenerateEngine(gen_path, tokenizer, local_only=local_only)

    raise RuntimeError(f"Failed to load inference engine for {gen_path}") from last_err

'''

PICK_MODEL_NAME = '''def pick_model_name() -> Tuple[str, str]:
    """Resolve DiffusionGemma checkpoint."""
    path = resolve_diffusiongemma_model_path()
    return path, "vllm-diffusion"

'''

RESOLVE_LOCAL = '''def resolve_local_model_path(prefer_generation: bool = True) -> Optional[str]:
    """Pick DiffusionGemma from explicit paths or Kaggle/HF cache."""
    explicit = [
        resolve_checkpoint_dir(KAGGLE_DIFFUSIONGEMMA_DIR),
        resolve_checkpoint_dir(LOCAL_DIFFUSIONGEMMA_DIR),
        resolve_checkpoint_dir(LOCAL_MODEL_PATH),
    ]
    for resolved in explicit:
        if resolved and local_dir_exists(resolved):
            return resolved
    discovered = discover_local_checkpoints()
    if discovered:
        for p in discovered:
            if "diffusiongemma" in p.lower():
                return p
        if prefer_generation:
            for p in discovered:
                if is_full_causal_lm_checkpoint(p):
                    return p
        return discovered[0]
    return None

'''

LOAD_MODELS = '''def load_models(model_name: str):
    """Load DiffusionGemma inference engine + tokenizer."""
    local_only = os.path.isdir(model_name) and local_dir_exists(model_name)
    gen_path = resolve_generation_model_path(model_name) if local_only else model_name
    load_label = gen_path if local_only else f"{gen_path} (remote)"
    print(
        f"\\n[LOAD] Initializing DiffusionGemma engine for {load_label}"
        + ("" if local_only else " (this can take 30-120s on first download)...")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        gen_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = create_inference_engine(gen_path, tokenizer)
    if isinstance(llm, HFGenerateEngine):
        setattr(llm, "model_path", gen_path)
    setattr(llm, "generation_model_path", gen_path)

    hf_model = None
    if isinstance(llm, HFGenerateEngine):
        hf_model = llm.model
    return llm, tokenizer, hf_model

'''

LOAD_LOCAL_MODELS = '''def load_local_models() -> Tuple[LLM, Any, Optional[LLM], Optional[Any]]:
    """Load local DiffusionGemma without network access."""
    print("\\n[LOCAL-LOAD] DiffusionGemma paths:")
    for tag, raw in (
        ("KAGGLE", KAGGLE_DIFFUSIONGEMMA_DIR),
        ("LOCAL", LOCAL_DIFFUSIONGEMMA_DIR),
        ("MODEL", LOCAL_MODEL_PATH),
    ):
        resolved = resolve_checkpoint_dir(raw) if raw else None
        print(f"    {tag:<6} {raw or '—'} -> {resolved or 'MISSING'}")
        if resolved:
            print_checkpoint_diagnostics(resolved)

    ensure_model_available()
    primary = resolve_local_model_path()
    if primary is None:
        raise RuntimeError(
            "[LOCAL-LOAD] No DiffusionGemma checkpoint found.\\n"
            f"  Add Kaggle Models input: google/diffusiongemma ({KAGGLE_DIFFUSIONGEMMA_DIR})"
        )

    gen_path = resolve_generation_model_path(primary)
    tokenizer = AutoTokenizer.from_pretrained(
        gen_path, trust_remote_code=True, local_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = create_inference_engine(
        gen_path,
        tokenizer,
        gpu_memory_utilization=DIFFUSION_GPU_UTIL,
    )
    if isinstance(llm, HFGenerateEngine):
        setattr(llm, "model_path", gen_path)
    setattr(llm, "generation_model_path", gen_path)
    backend = "HF-fallback" if isinstance(llm, HFGenerateEngine) else "vLLM"
    print(f"[LOCAL-LOAD] {os.path.basename(gen_path)} engine ready ({backend}).")
    return llm, tokenizer, None, None

'''

VOTER_POOL_GEN = '''def _resolve_voter_pool_gen_path() -> str:
    """Resolve DiffusionGemma checkpoint for voter pool workers."""
    path = resolve_diffusiongemma_model_path()
    print(f"[VOTER-POOL] Generation checkpoint: {path}", flush=True)
    return os.path.abspath(path)

'''


def delete_ranges(lines: list[str], ranges: list[tuple[int, int]]) -> list[str]:
    drop = set()
    for start, end in ranges:
        for i in range(start, end + 1):
            drop.add(i)
    return [line for i, line in enumerate(lines, start=1) if i not in drop]


def replace_function_block(text: str, name: str, replacement: str) -> str:
    pattern = rf"(?ms)^def {re.escape(name)}\(.*?\n(?=^def |^class |^@dataclass|^# ={10,}|^if __name__)"
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"function not found: {name}")
    return text[: m.start()] + replacement + text[m.end() :]


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")

    # Replace file header up to toggles section
    src = re.sub(r"(?s)^#!/usr/bin/env python3.*?# ={10,}\n# TOGGLES", NEW_HEADER + "# =============================================================================\n# TOGGLES", src, count=1)

    for old, new in REPLACEMENTS:
        if old in src:
            src = src.replace(old, new, 1)
        elif old.strip():
            print(f"WARN: replacement anchor missing ({old[:60]}...)")

    tree = ast.parse(src)
    ranges = list(EXTRA_RANGES)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in REMOVE_FUNCS:
            ranges.append((node.lineno, node.end_lineno))
        if isinstance(node, ast.ClassDef) and node.name in REMOVE_CLASSES:
            ranges.append((node.lineno, node.end_lineno))

    lines = delete_ranges(src.splitlines(keepends=True), sorted(ranges))
    text = "".join(lines)

    # Replace simplified blocks
    text = replace_function_block(text, "million_brains_dflash_generate", MILLION_BRAINS_GENERATE)
    text = replace_function_block(text, "resolve_generation_model_path", RESOLVE_GENERATION_MODEL_PATH)
    text = re.sub(
        r"(?ms)# ={10,}\n# REQUIRED BANNER.*?\nif not _MBR_WORKER_SUBPROCESS:\n    print_one_million_brains_banner\([^)]+\)\n\n",
        BANNER_BLOCK,
        text,
        count=1,
    )
    text = replace_function_block(text, "_adaptive_vllm_gpu_util", ADAPTIVE_GPU_UTIL)
    text = replace_function_block(text, "_build_vllm_attempts", BUILD_VLLM_ATTEMPTS)
    text = replace_function_block(text, "_build_vllm_worker_fast_attempts", BUILD_WORKER_ATTEMPTS)
    text = replace_function_block(text, "create_inference_engine", CREATE_INFERENCE_ENGINE)
    text = replace_function_block(text, "pick_model_name", PICK_MODEL_NAME)
    text = replace_function_block(text, "resolve_local_model_path", RESOLVE_LOCAL)
    text = replace_function_block(text, "load_models", LOAD_MODELS)
    text = replace_function_block(text, "load_local_models", LOAD_LOCAL_MODELS)
    text = replace_function_block(text, "_resolve_voter_pool_gen_path", VOTER_POOL_GEN)

    # HF engine: drop dflash param
    text = text.replace("dflash_draft_path: Optional[str] = None,\n", "")
    text = text.replace("        dflash_draft_path: Optional[str] = None,\n", "")
    text = text.replace("        if is_dflash_draft_checkpoint(model_path):\n            raise ValueError(\n                f\"Refusing to load DFlash draft as causal LM: {model_path}\"\n            )\n\n", "")
    text = re.sub(
        r"        if dflash_draft_path:.*?fallback cannot run vLLM speculative decoding\.\n            \)\n",
        "",
        text,
        flags=re.S,
    )
    text = text.replace("        self.dflash_draft_path = dflash_draft_path\n", "")
    text = text.replace(
        '                "this is likely a DFlash draft checkpoint, not a base causal LM."',
        '"checkpoint has no usable embed_tokens."',
    )
    text = text.replace(
        '    """Fallback engine when vLLM cannot load a custom DFlash / Qwen3.5 checkpoint."""',
        '    """Fallback engine when vLLM cannot load DiffusionGemma."""',
    )

    # verify_inference_engine cleanup
    text = re.sub(
        r"    DFlash drafts mistakenly loaded as causal LMs\.\n    \"\"\"",
        '    """',
        text,
    )
    text = re.sub(r"    draft_path = getattr\(llm, \"dflash_draft_path\", None\)\n.*?print\(f\"    spec_dec.*?\n", "", text, flags=re.S)
    text = text.replace(
        '            "and DFlash speculative decoding is NOT used. Fix vLLM load or set "',
        '"Fix vLLM load or set "',
    )
    text = text.replace(
        '                f"spec={getattr(llm, \'speculative_decoding_enabled\', False)}"',
        '"spec=n/a"',
    )
    text = re.sub(
        r'                "\\n  Likely cause: DFlash draft loaded as causal LM, or stale script "',
        '"\\n  Likely cause: stale script "',
        text,
    )
    text = text.replace(
        '            "Load tokenizer from the paired BASE model, not the DFlash draft."',
        '"Tokenizer produced empty input_ids."',
    )

    # checkpoint priority
    text = text.replace(
        '    """When a mount nests BASE + DFlash, prefer the full causal-LM checkpoint."""',
        '    """Prefer full causal-LM checkpoints when multiple mounts exist."""',
    )
    text = re.sub(r"    draft_rank = 1 if is_dflash_draft_checkpoint\(path\) else 0\n", "    draft_rank = 0\n", text)
    text = text.replace(
        '    """Lower tuple sorts earlier: prefer DFlash-tuned, then larger Qwen, then lexicographic."""',
        '    """Lower tuple sorts earlier: prefer larger models, then lexicographic."""',
    )
    text = re.sub(r'    if "dflash" in name:\n        score -= 100\n', "", text)
    text = re.sub(
        r'            role = "FULL" if is_full_causal_lm_checkpoint\(p\) else \(\n                "DRAFT" if is_dflash_draft_checkpoint\(p\) else "UNKNOWN"\n            \)',
        '            role = "FULL" if is_full_causal_lm_checkpoint(p) else "UNKNOWN"',
        text,
    )

    # bundle_folder_role simplify
    text = re.sub(
        r'(?ms)def bundle_folder_role\(model_path: str\) -> Optional\[str\]:.*?return "base"',
        '''def bundle_folder_role(model_path: str) -> Optional[str]:
    """Classify checkpoint folder role (DiffusionGemma = base generation)."""
    if not model_path:
        return None
    name = os.path.basename(model_path).lower()
    if "diffusiongemma" in name or "diffusion" in name:
        return "base"
    if is_full_causal_lm_checkpoint(model_path):
        return "base"''',
        text,
        count=1,
    )

    # benchmark / main strings
    text = text.replace(
        '        "MILLION-BRAINS-DIFFUSIONGEMMA"\n        if USE_DIFFUSIONGEMMA\n        else "MILLION-BRAINS-DFLASH"',
        '"MILLION-BRAINS-DIFFUSIONGEMMA"',
    )
    text = re.sub(r'    print\(f"vLLM speculative: \{.*?\}\)\n', "", text)
    text = text.replace(
        "\n[MBR] Running full one-million-brains-dflash (permutation allocator + CTSB smoothing + cross-stream integration + adaptive reallocation) ...",
        "\n[MBR] Running Million-Brains DiffusionGemma conditioned denoising ...",
    )
    text = re.sub(
        r'        f"    ENGINE: diffusiongemma=\{USE_DIFFUSIONGEMMA\} .*?f"phase1_parallel=\{ARC_PHASE1_PROMPT_PARALLELISM\}"\n    \)',
        '        f"    ENGINE: DiffusionGemma canvas={DIFFUSION_CANVAS_LENGTH} denoise_k={K} | "\n        f"spatial_ensemble={ARC_SPATIAL_GRID_ENSEMBLE} "\n        f"phase1_parallel={ARC_PHASE1_PROMPT_PARALLELISM}"\n    )',
        text,
        flags=re.S,
    )

    # stray references
    for token in ("USE_DIFFUSIONGEMMA", "ENABLE_DFLASH_LIVE_PATCH", "ENABLE_VLLM_SPECULATIVE_DECODING",
                  "is_dflash_draft_checkpoint", "discover_qwen_dflash_bundle_paths",
                  "find_paired_dflash_draft", "find_paired_base_checkpoint",
                  "_inference_speculative_status", "_vllm_spec_attempt_should_abort",
                  "VLLM_REQUIRE_SPECULATIVE", "VLLM_SPEC_", "dflash_draft_path",
                  "KAGGLE_QWEN_BUNDLE_ROOT", "LOCAL_DFLASH_DIR", "DRAFT_BUNDLE"):
        if token in text:
            print(f"WARN: leftover token: {token}")

    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {TARGET} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()