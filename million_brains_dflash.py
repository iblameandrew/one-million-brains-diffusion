#!/usr/bin/env python3
"""
million_brains_dflash.py — ONE-MILLION-BRAINS-DIFFUSIONGEMMA
Permutation-Gated Feature-Slot Allocator hard-wired into DiffusionGemma canvas denoising.

Architecture: DiffusionGemma block-diffusion (256-token canvas, iterative denoising) with
K parallel Million-Brains conditioned trajectories per denoise step — feature-slot allocation,
CTSB smoothing, cross-stream integration, cumprod verification, and adaptive reallocation.

Kaggle: attach Models input google/diffusiongemma, run as script or notebook cell.
"""

# =============================================================================
# TOGGLES - ALL USER CONTROLS LIVE HERE (edit and re-run)
# =============================================================================
SCRIPT_VERSION = "2026-06-19-diffusion-e"  # batched Phase-1 + tokenization hot-path fixes
# --- DiffusionGemma core (default engine) ---
# Kaggle: Add Input -> Models -> google/diffusiongemma -> diffusiongemma-26b-a4b-it
KAGGLE_DIFFUSIONGEMMA_DIR = (
    "/kaggle/input/models/google/diffusiongemma/transformers/diffusiongemma-26b-a4b-it/1"
)
LOCAL_DIFFUSIONGEMMA_DIR = KAGGLE_DIFFUSIONGEMMA_DIR  # override for local dev if needed
DIFFUSIONGEMMA_MODEL_PRIMARY = "google/diffusiongemma-26B-A4B-it"  # HF id (offline fallback)
DIFFUSIONGEMMA_MODEL_FALLBACK = "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"  # NVFP4 variant
DIFFUSION_CANVAS_LENGTH = 256  # vLLM diffusion_config canvas block size
DIFFUSION_MAX_NUM_SEQS = 4  # vLLM recipe: keep low (diffusion state ∝ max_seqs × canvas × vocab)
DIFFUSION_ENTROPY_BOUND = 0.1  # entropy-bound denoising sampler
DIFFUSION_GPU_UTIL = 0.85  # headroom for denoising activations on H100/A100/L4
DIFFUSION_MAX_MODEL_LEN = 8192  # ARC + chat; raise on A100 if VRAM allows
DIFFUSION_DENOISE_CHUNK = 6  # tokens committed per denoise step (= legacy BLOCK_SIZE)
K = 4  # Million-Brains parallel trajectories per denoise step (not ARC hypothesis count)
ARC_HYPOTHESIS_SLOTS = 8  # ARC eval: batched hypothesis proposals per engine (vLLM shows N/N)
NUM_PERSONALITY_FEATURES = 12  # personality / spatial primitive bank for allocator
ARC_FORCE_ENABLE_THINKING = False  # Step 2: never emit Qwen3.5 </think> chains in ARC
ARC_GUIDED_JSON_DECODING = True  # Step 3: vLLM guided JSON for grid outputs
ARC_SPATIAL_GRID_ENSEMBLE = True  # Step 4: Phase1=8 grid hypotheses, Phase2=pixel majority vote
BLOCK_SIZE = DIFFUSION_DENOISE_CHUNK  # tokens committed per denoise step / super-block
MAX_SUPERBLOCKS = 32  # safety cap on super-blocks
ENABLE_FEATURE_REALLOCATION = True  # master switch for adaptive reallocation of features into slots based on acceptance
ACCEPTANCE_THRESHOLD = 0.28  # below this a feature-slot is considered underperforming and eligible for reallocation
REFRAME_TEMP_BOOST = 0.35  # additive temperature boost on total super-block rejection
BASE_TEMPERATURE = 0.7
BASE_TOP_P = 0.92
TARGET_MAX_TOKENS = 160  # benchmark generation length
BENCHMARK_PROMPT = (  # a single hard prompt that benefits from combinatorial diversity of personality features
    "You are a world-class puzzle solver. Think step-by-step with extreme rigor. "
    "A farmer has 7 chickens, 4 pigs, and 3 cows. Each chicken has 2 legs, pigs have 4, cows have 4. "
    "At exactly noon every animal casts a perfect shadow that an observer might mistakenly count as extra legs. "
    "The observer counts 61 'legs' total (real + shadow). How many legs are actually real? "
    "Explain your reasoning in numbered steps and give the final integer answer."
)
SEED = (
    42  # for reproducibility of permutation hashing + sampling inside active features
)
ENABLE_CIRCUIT_SMOOTHING = True  # CTSB: smooth transitions between permutation circuits
CTSB_BLEND_TAU = 0.35  # acceptance-gated blend time constant (higher = slower circuit morph)
CTSB_MAX_SLOT_SWAPS = 2  # max feature-slot changes per super-block (geodesic step)
CTSB_DSB_EMA = 0.82  # discourse state buffer EMA decay
CTSB_COHERENCE_ALPHA = 0.55  # TAFK weight on cumprod acceptance rate
CTSB_COHERENCE_GAMMA = 0.25  # TAFK weight on discourse coherence
CTSB_COHERENCE_ETA = 0.15  # TAFK penalty for stylistic jump vs committed prefix
CTSB_COHERENCE_KAPPA = 0.12  # TAFK bonus for features continuing from previous circuit
ANCHOR_SLOT = 0  # slot 0 receives extra smoothing inertia (stable chain scribe)
# Local/offline model paths (Kaggle: attach a HF dataset or pre-download to /kaggle/working)
PREFER_LOCAL_MODELS = True  # try /kaggle/input + /kaggle/working before any HuggingFace request
# Local model path override (optional; DiffusionGemma resolved via KAGGLE_DIFFUSIONGEMMA_DIR)
LOCAL_MODEL_PATH = KAGGLE_DIFFUSIONGEMMA_DIR
VLLM_ENFORCE_EAGER = False  # prefer CUDA graphs; load loop retries enforce_eager=True on failure
VLLM_RUNNER = "generate"  # do not let vLLM pick pooling/embedding runner
VLLM_FALLBACK_TO_HF = True  # HuggingFace wrapper if vLLM still cannot load the checkpoint
VLLM_GPU_MEMORY_UTILIZATION = 0.88  # HF fallback only; DiffusionGemma uses DIFFUSION_GPU_UTIL
VLLM_TENSOR_PARALLEL_SIZE = 0  # 0=auto (retry with tp=2 when 2+ GPUs visible)
PREFER_HF_INFERENCE = False  # L4/A10+: prefer vLLM fast kernels; HF only on load failure
SKIP_VLLM_FOR_QWEN35 = False  # attempt vLLM for Qwen3.5-4B (set True on broken vLLM hosts)
AUTO_PREFETCH_TO_WORKING = True  # when online, try to cache DiffusionGemma into /kaggle/working
PREFETCH_MODEL_ID = DIFFUSIONGEMMA_MODEL_PRIMARY
KAGGLEHUB_MODEL_HANDLE = ""  # set if you publish DiffusionGemma on Kaggle Models
KAGGLE_DATASET_HANDLE = ""  # optional dataset fallback on Kaggle
# ---------------------------------------------------------------------------
# ARC DATA — copy-paste into Kaggle:
#   1) Add competition input "arc-prize-2026-arc-agi-2"
#   2) Either keep ARC_DATA_PROFILE = "auto" (detects Kaggle mount automatically)
#      or force: ARC_DATA_PROFILE = "kaggle"
# Local dev: auto picks data/ when present; or set ARC_DATA_PROFILE = "local"
# ---------------------------------------------------------------------------
ARC_DATA_PROFILE = "auto"  # "auto" | "kaggle" | "local" | "off"
ARC_DATA_SPLIT = "evaluation"  # "training" | "evaluation"
KAGGLE_ARC_COMPETITION_DIR = (
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
)
LOCAL_ARC_DATA_DIR = "data"
# Optional explicit overrides (None = derived from profile + split above)
EVAL_CHALLENGES_PATH = None
EVAL_SOLUTIONS_PATH = None
EVAL_MAX_TASKS = None  # cap tasks for smoke tests; None = all tasks in challenges file
EVAL_MAX_NEW_TOKENS = 512  # per-task generation budget for ARC eval
ARC_VISUAL_GRADING = True  # print + display a grade card for every test case
ARC_PRINT_ALL_ANSWERS = True  # print every prediction vs ground-truth JSON for every test case
# Real-time streaming — prints every token and all model/MBR activity as it happens
STREAM_GENERATION = True  # stream decoded tokens to stdout live (flush per token)
STREAM_ALL_OUTPUT = True  # print all MBR drafts, commits, realloc, verify steps
STREAM_PRINT_THINKING = False  # Step 2: ARC uses enable_thinking=False (no internal monologue)
STREAM_VERIFY_PASSES = False  # verify=logprob-only; never stream those tokens
HF_FAST_VERIFY = True  # tail-only logprobs on draft suffix (not full 4k prompt)
ARC_EVAL_VERBOSE = True  # MBR super-block internals (auto-on when STREAM_ALL_OUTPUT)
ARC_USE_CHAT_TEMPLATE = True  # Qwen3.5 expects chat_template.jinja wrapping
ARC_DISABLE_THINKING = True  # Step 2: hard-off thinking for ARC (overrides STREAM_PRINT_THINKING)
ARC_STRUCTURED_THINKING = True  # require incremental HYPOTHESIS_GRID inside <think>
ARC_PRINT_STEP_MATRICES = False  # debug: ASCII grids at every MBR draft slot + commit
ARC_PRINT_FINAL_MATRICES = True  # one final ASCII summary per test (gold + preds + verdict + timing)
ARC_ASSISTANT_PREFILL = "[["  # chat prefill anchors JSON grid (skipped when thinking is on)
ARC_GENERATION_TEMPERATURE = 0.0  # greedy JSON for grid tasks
EVAL_SMOKE_TASK_ID = None  # e.g. "0934a4d8" — run only this task (overrides max_tasks slice)
ARC_FAST_INFERENCE = True  # ARC eval: compact prompts, batched gen, no per-token streaming
ARC_TRY_VLLM = True  # ARC eval requires vLLM for speed; HF fallback if load fails
# Qwen3.5-4B native context is 150k+ (YaRN); vLLM max_model_len is a VRAM budget we set at load.
# KV cache grows with max_model_len — 4k was only to fit L4 OOM; ARC 30x30 tasks need ~8k+.
VLLM_MAX_MODEL_LEN = 0  # 0 = auto (ARC prompt + output budget); set e.g. 16384 if VRAM allows
ARC_MAX_PROMPT_TOKENS = 6144  # tighter input budget; resolver shrinks further; lowers KV bandwidth on decode
ARC_SLOT_HYPOTHESIS_MODE = True  # ARC eval uses slot pipeline (spatial grids when ARC_SPATIAL_GRID_ENSEMBLE)
ARC_MBR_OUTPUT_TOKEN_BUDGET = 14000  # total OUTPUT tokens per test (hyp slots + final grid combined)
ARC_FINAL_GRID_MIN_TOKENS = 512  # floor for final JSON grid pass
ARC_FINAL_GRID_MAX_FRACTION = 0.85  # final may use up to 85% of output budget for large grids
ARC_FINAL_GRID_MIN_FRACTION = 0.50  # always reserve 50% of output budget for final grid synthesis
ARC_HYPOTHESIS_MAX_TOKENS = ARC_MBR_OUTPUT_TOKEN_BUDGET * 3 // 4 // 8  # legacy default; task-aware below
ARC_FINAL_GRID_MAX_TOKENS = ARC_MBR_OUTPUT_TOKEN_BUDGET // 4  # legacy default; task-aware below
ARC_FINAL_HYP_CHAR_CAPS = (360, 200, 120, 60, 0)  # shrink hypothesis text in final prompt if needed
ARC_HYPOTHESIS_ENABLE_THINKING = False  # False = fast text rules (True + thinking can stall tqdm at 0/N for minutes)
ARC_HYPOTHESIS_THINKING_TOKEN_CAP = 512  # per-slot cap when hypothesis thinking is enabled
ARC_FINAL_ENABLE_THINKING = False  # final grid: [[ prefill + greedy JSON (much faster than thinking pass)
ARC_PHASE1_PROMPT_PARALLELISM = True  # batch K slots per vLLM generate() (use_tqdm=False avoids stall)
ARC_VLLM_MAX_NUM_SEQS = 4  # Phase-1 batch size; raise to 8 on A100+ if VRAM allows
ARC_CHAT_TEMPLATE_SLACK = 2048  # system + chat template overhead beyond raw task body
ARC_SAVE_GRADE_IMAGES = True  # save PNG grade cards (arc_grades/ or /kaggle/working/arc_grades/)
ARC_SHOW_TRAIN_EXAMPLES = True  # include train pair thumbnails on each grade card
ARC_ANSWER_REPORT_PATH = None  # None = auto (/kaggle/working/arc_answer_report.json or arc_answer_report.json)

# =============================================================================
# STANDARD LIBRARY + ML IMPORTS
# =============================================================================
import os
import sys
import socket
import subprocess
import inspect
import argparse
import re
import math
import time
import random
import json
import hashlib
import traceback
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter

import warnings
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _suppress_noisy_third_party_logs() -> None:
    """Kaggle notebooks spam transformers/vLLM deprecation lines every worker spawn."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings(
        "ignore",
        message=r".*Qwen2VLImageProcessorFast.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Fast.* suffix for image processors.*",
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"cuda.*")
    for logger_name in (
        "transformers",
        "transformers.image_processing_utils",
        "transformers.processing_utils",
        "vllm",
        "vllm.model_executor",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


_suppress_noisy_third_party_logs()

# Heavy libraries (installed above)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams

# =============================================================================
# FIXED BANK OF 12 PERSONALITY FEATURES
# Step 4: 12 spatial transformation primitives (replaces personality personas for ARC).
# Allocator still permutes K primitives into parallel slots for the demo/benchmark path.
# =============================================================================
SPATIAL_PRIMITIVES: List[str] = [
    "Rotate90",
    "Rotate180",
    "ReflectH",
    "ReflectV",
    "Transpose",
    "CropBBox",
    "TileRepeat",
    "ColorMap",
    "SymmetryComplete",
    "FloodFill",
    "ComponentExtract",
    "GravityShift",
]

# Legacy alias — PermutationFeatureSlotAllocator and benchmark code use this name.
PERSONALITY_FEATURES: List[str] = SPATIAL_PRIMITIVES

# Greedy grid decoding for spatial ensemble; benchmark may still vary temperature.
FEATURE_PARAMS: Dict[str, Dict[str, float]] = {
    name: {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.02}
    for name in SPATIAL_PRIMITIVES
}

# Per-slot spatial lens: Phase 1 emits a JSON grid hypothesis (not prose).
SPATIAL_PRIMITIVE_LENSES: Dict[str, str] = {
    "Rotate90": "Apply a 90-degree clockwise rotation rule inferred from train pairs.",
    "Rotate180": "Apply a 180-degree rotation rule inferred from train pairs.",
    "ReflectH": "Apply horizontal reflection (mirror left-right) across train examples.",
    "ReflectV": "Apply vertical reflection (mirror top-bottom) across train examples.",
    "Transpose": "Apply matrix transpose / axis-swap patterns seen in train outputs.",
    "CropBBox": "Crop to the minimal bounding box of non-background objects, then resize.",
    "TileRepeat": "Detect tiling/repetition: repeat or scale a motif to output size.",
    "ColorMap": "Infer a deterministic color permutation (0-9) mapping input cells to output.",
    "SymmetryComplete": "Complete partial symmetries (mirror/rotate) to form the output grid.",
    "FloodFill": "Flood-fill enclosed regions or propagate a dominant color from seeds.",
    "ComponentExtract": "Extract connected components, relabel, and place into output layout.",
    "GravityShift": "Shift non-zero cells down/left as if gravity acts on colored pixels.",
}

# Back-compat alias for text-hypothesis path (unused when ARC_SPATIAL_GRID_ENSEMBLE=True).
SLOT_HYPOTHESIS_LENSES: Dict[str, str] = SPATIAL_PRIMITIVE_LENSES

assert len(PERSONALITY_FEATURES) == NUM_PERSONALITY_FEATURES, (
    "PERSONALITY_FEATURES list length must equal NUM_PERSONALITY_FEATURES"
)


# =============================================================================
# COMBINATORIAL PERMUTATION UNRANKING (mathematically rigorous hashing)
# =============================================================================
def calc_permutation_count(n: int, k: int) -> int:
    """P(n, k) = n! / (n-k)!   (number of injective functions from k to n)"""
    p = 1
    for i in range(k):
        p *= n - i
    return p


def unrank_permutation(rank: int, n: int, k: int) -> List[int]:
    """
    Decode `rank` (0 <= rank < P(n,k)) into an ordered k-tuple of distinct indices in [0, n).

    Uses the factorial number system (combinadic / Lehmer code variant).
    This is the core primitive that turns a hash seed into a concrete assignment
    of personality features to the K feature-slots.

    Example (n=12, k=3):
        rank 0            -> [0, 1, 2]
        rank 1            -> [0, 1, 3]
        ...
        rank = 12*11*10-1 -> [11, 10, 9]
    """
    if k == 0:
        return []
    p = calc_permutation_count(n, k)
    rank = rank % p

    used = list(range(n))
    result: List[int] = []
    for i in range(k):
        remaining = n - i
        f = math.factorial(remaining - 1) if (remaining - 1) > 0 else 1
        idx = (rank // f) % remaining
        result.append(used.pop(idx))
        rank %= f
    return result


def hash_to_feature_permutation(
    pooled_vec: torch.Tensor, step: int, n: int, k: int
) -> List[int]:
    """
    Core of the permutation-focused allocator.

    Turn a pooled hidden vector (from previous target verification) + super-block step
    into a deterministic ordered k-tuple of distinct personality feature indices.

    This is a pure combinatorial unranking:
      1. Derive a stable 64-bit seed from the pooled vector (mean + variance moments) + step.
      2. rank = seed % P(n, k)
      3. Decode rank with the factorial number system (combinadic) into an injection
         of length k with no repeats.

    The returned list is directly the assignment of personality features to feature-slots
    0..k-1 for the upcoming super-block.
    """
    if pooled_vec is None or pooled_vec.numel() == 0:
        seed = step * 0x9E3779B97F4A7C15
    else:
        v = pooled_vec.detach().float().flatten()
        mean_v = v.mean().item()
        var_v = (v - mean_v).pow(2).mean().item() + 1e-12
        h = (
            int((mean_v * 1_000_003 + var_v * 1_000_037 + step * 37) * 1_000_000_007)
            & 0xFFFFFFFFFFFFFFFF
        )
        seed = h

    p = calc_permutation_count(n, k)
    rank = seed % p
    return unrank_permutation(rank, n, k)


# =============================================================================
# PERMUTATION-GATED FEATURE-SLOT ALLOCATOR
# =============================================================================
class PermutationFeatureSlotAllocator(nn.Module):
    """
    The central combinatorial component.

    Input: pooled hidden state after a target verification forward (the "memory" of
           what the model just accepted or rejected).
    Output: a fresh permutation of K distinct personality features chosen from the
            fixed bank of 12. These are assigned to the K feature-slots for the next
            super-block of parallel drafting.

    Mathematically this is exactly a deterministic selection of an element from the
    set of all possible injections [0..11] choose K, ordered, via modular unranking.

    The selected features are injected via:
        - feature_emb lookup (added to hidden states of the corresponding stream)
        - feature-specific sampling parameters (temperature / top_p) for proposal
        - per-feature gating vectors (would be applied after the cross-stream integration
          attention in a real kernel implementation)

    In this script the allocator is also used to drive adaptive reallocation: when a
    slot's recent acceptance is poor we simply let the next hash-based permutation
    bring in a different feature (or explicitly swap from the unused pool).
    """

    def __init__(self, internal_dim: int = 256, num_features: int = 12, k: int = 4):
        super().__init__()
        self.num_features = num_features
        self.k = k
        self.internal_dim = internal_dim

        # Tiny optional refiner MLP (the actual selection remains the hash permutation;
        # this MLP exists only for educational symmetry with "gated" designs).
        self.seed_refiner = nn.Sequential(
            nn.Linear(internal_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # The 12 personality feature embedding vectors.
        # In a full low-level patch these would be added (after the normal token embedding)
        # or used to produce per-stream bias / scale inside the drafter forward.
        self.feature_emb = nn.Embedding(num_features, internal_dim)

        # Per-feature gating vectors (post cross-stream integration scaling / bias).
        self.feature_gates = nn.Parameter(torch.ones(num_features, internal_dim) * 0.9)

        # Positional components used to differentiate the K feature-slotted streams.
        self.segment_emb = nn.Embedding(64, internal_dim)  # super-block / horizon index
        self.stream_pos_emb = nn.Embedding(
            k, internal_dim
        )  # which feature-slot / stream lane

        self._p = calc_permutation_count(num_features, k)

    def forward(self, pooled: torch.Tensor, step: int) -> Dict[str, Any]:
        """
        Returns a dict describing the allocation for this super-block:
            feature_indices: List[int] length K   -- the personality feature ids assigned to slots 0..K-1
            feature_vectors: [K, D]               -- the corresponding feature_emb rows
            gates: [K, D]                         -- the per-feature gate vectors
            stream_pos: [K, D]                    -- lane offsets for the K streams
            feature_names: List[str]              -- human-readable labels of the active features
        """
        feature_indices = hash_to_feature_permutation(
            pooled, step, self.num_features, self.k
        )

        # Optional neural adjustment of the seed (still fully deterministic given pooled + step).
        if pooled is not None and pooled.numel() > 0:
            v = pooled.detach().float().flatten()
            if v.shape[0] < self.internal_dim:
                pad = torch.zeros(self.internal_dim - v.shape[0], device=v.device)
                v = torch.cat([v, pad])
            else:
                v = v[: self.internal_dim]
            v = v.unsqueeze(0)
            _ = self.seed_refiner(v)

        feature_vectors = self.feature_emb(
            torch.tensor(feature_indices, dtype=torch.long)
        )
        gates = self.feature_gates[torch.tensor(feature_indices, dtype=torch.long)]
        stream_pos = self.stream_pos_emb(torch.arange(self.k))

        return {
            "feature_indices": feature_indices,
            "feature_vectors": feature_vectors,
            "gates": gates,
            "stream_pos": stream_pos,
            "feature_names": [PERSONALITY_FEATURES[i] for i in feature_indices],
        }

    def get_feature_params(self, feature_indices: List[int]) -> List[Dict[str, float]]:
        """Return the FEATURE_PARAMS (temperature/top_p etc.) for each allocated feature."""
        return [FEATURE_PARAMS[PERSONALITY_FEATURES[fid]] for fid in feature_indices]


# =============================================================================
# FAKE / SIMULATED POOLED HIDDEN STATE (from text history)
# Educational stand-in for "target_hidden" after verification step.
# =============================================================================
def make_pooled_state(
    generated_ids: List[int],
    step: int,
    dim: int = 256,
    tokenizer: Optional[Any] = None,
) -> torch.Tensor:
    """
    Produce a cheap but history-dependent vector that the orchestrator can hash.
    In a true low-level patch this would be the mean-pooled last-layer hidden state
    of the target model right after the verification forward (rolling target_hidden).
    """
    if not generated_ids:
        base = torch.tensor([float((step * 17 + 3) % 97)], dtype=torch.float32)
    else:
        tail = generated_ids[-16:] if len(generated_ids) > 16 else generated_ids
        # Mix token ids with positional information (simulates "offset KV advance")
        vals = []
        for i, tid in enumerate(tail):
            vals.append(((tid % 503) + (i * 7) + (step * 13)) * 0.013)
        base = torch.tensor(vals, dtype=torch.float32)

    # Expand / fold into fixed internal_dim
    out = torch.zeros(dim, dtype=torch.float32)
    for i, v in enumerate(base):
        out[(i * 17) % dim] += v
    # Add a tiny amount of step-dependent structured noise (so different blocks explore)
    out[(step * 3) % dim] += 0.07 * ((step % 7) - 3)
    return out.unsqueeze(0)  # [1, dim]


# =============================================================================
# CIRCUIT TRANSITION SMOOTHING BLOCK (CTSB)
# Smooths super-block handoffs between permutation circuits while preserving
# semantic coherence of the committed thinking chain.
# =============================================================================
@dataclass
class CTSBState:
    """Persistent cross-circuit memory carried across super-blocks."""

    discourse: Optional[torch.Tensor] = None
    prev_feature_indices: Optional[List[int]] = None
    prev_params: Optional[List[Dict[str, float]]] = None
    prev_committed_ids: List[int] = field(default_factory=list)
    lambda_history: List[float] = field(default_factory=list)
    effective_feature_history: List[List[str]] = field(default_factory=list)


class CircuitTransitionSmoother:
    """
    Inter-super-block smoothing layer between the permutation allocator and draft phase.

    Subsystems (v1 Python-side):
      - DSB: discourse state buffer (slow semantic memory)
      - CBF/SPI: blend feature vectors and sampling params toward new circuit
      - Geodesic slot step: limit abrupt feature identity jumps
      - TAFK: transition-aware fusion kernel (stream-level coherent selection)
    """

    def __init__(
        self,
        k: int = K,
        num_features: int = NUM_PERSONALITY_FEATURES,
        internal_dim: int = 256,
        allocator: Optional[PermutationFeatureSlotAllocator] = None,
    ):
        self.k = k
        self.num_features = num_features
        self.internal_dim = internal_dim
        self.allocator = allocator
        self.state = CTSBState()

    def _default_params(self) -> Dict[str, float]:
        return {
            "temperature": BASE_TEMPERATURE,
            "top_p": BASE_TOP_P,
            "repetition_penalty": 1.03,
        }

    def _params_to_vec(self, params: Dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [
                params.get("temperature", BASE_TEMPERATURE),
                params.get("top_p", BASE_TOP_P),
                params.get("repetition_penalty", 1.03),
            ],
            dtype=torch.float32,
        )

    def _vec_to_params(self, vec: torch.Tensor, template: Dict[str, float]) -> Dict[str, float]:
        out = dict(template)
        out["temperature"] = float(max(0.05, vec[0].item()))
        out["top_p"] = float(min(1.0, max(0.05, vec[1].item())))
        out["repetition_penalty"] = float(max(1.0, vec[2].item()))
        return out

    def circuit_distance(self, prev: List[int], curr: List[int]) -> int:
        return sum(1 for a, b in zip(prev, curr) if a != b)

    def geodesic_feature_indices(
        self, prev: List[int], target: List[int], max_swaps: int
    ) -> List[int]:
        """Move at most max_swaps slot assignments toward the allocator target."""
        if not prev or len(prev) != len(target):
            return list(target)
        result = list(prev)
        swaps = 0
        for slot in range(self.k):
            if swaps >= max_swaps:
                break
            if result[slot] != target[slot]:
                result[slot] = target[slot]
                swaps += 1
        return result

    def _extract_step_meta(self, text: str) -> Dict[str, float]:
        steps = [int(m.group(1)) for m in re.finditer(r"(?:Step|step)\s*(\d+)", text)]
        last_step = float(steps[-1]) if steps else 0.0
        has_numbered = 1.0 if steps else 0.0
        return {"last_step": last_step, "has_numbered": has_numbered}

    def update_discourse(
        self,
        pooled: torch.Tensor,
        generated_ids: List[int],
        accepted_ids: List[int],
        committed_text: str,
    ) -> torch.Tensor:
        """DSB: EMA of pooled verification state + lexical step structure."""
        dim = self.internal_dim
        h_verify = pooled.detach().float().flatten()
        if h_verify.numel() < dim:
            h_verify = F.pad(h_verify, (0, dim - h_verify.numel()))
        else:
            h_verify = h_verify[:dim]

        tail = accepted_ids[-12:] if accepted_ids else generated_ids[-12:]
        h_lex = torch.zeros(dim, dtype=torch.float32)
        for i, tid in enumerate(tail):
            h_lex[(tid + i * 13) % dim] += 0.11

        meta = self._extract_step_meta(committed_text)
        h_meta = torch.zeros(dim, dtype=torch.float32)
        h_meta[0] = meta["has_numbered"]
        h_meta[1] = meta["last_step"] * 0.05

        instant = 0.55 * h_verify + 0.30 * h_lex + 0.15 * h_meta
        if self.state.discourse is None:
            self.state.discourse = instant.clone()
        else:
            self.state.discourse = (
                CTSB_DSB_EMA * self.state.discourse + (1.0 - CTSB_DSB_EMA) * instant
            )
        return self.state.discourse

    def compute_blend_lambda(
        self,
        acceptance_rate: float,
        accepted_len: int,
        circuit_dist: int,
        full_rejection: bool,
    ) -> float:
        """How far into the new circuit the upcoming block should lean (0=prev, 1=target)."""
        if full_rejection:
            return 1.0
        accept_gate = 1.0 - math.exp(-max(accepted_len, 0) / max(CTSB_BLEND_TAU, 1e-6))
        proximity = 1.0 - min(1.0, circuit_dist / max(1, self.k))
        lam = accept_gate * (0.35 + 0.65 * proximity)
        if acceptance_rate > 0.5:
            lam *= 0.60
        elif acceptance_rate > 0.35:
            lam *= 0.78
        return float(max(0.12, min(1.0, lam)))

    def smooth_sampling_params(
        self,
        prev_params: List[Dict[str, float]],
        target_params: List[Dict[str, float]],
        blend_lambda: float,
    ) -> List[Dict[str, float]]:
        """SPI: interpolate per-slot temperature / top_p / repetition_penalty."""
        smoothed: List[Dict[str, float]] = []
        for slot in range(self.k):
            prev = prev_params[slot] if slot < len(prev_params) else self._default_params()
            target = target_params[slot]
            anchor_boost = 0.55 if slot == ANCHOR_SLOT else 1.0
            lam = min(1.0, blend_lambda * anchor_boost)
            pv = self._params_to_vec(prev)
            tv = self._params_to_vec(target)
            blended = (1.0 - lam) * pv + lam * tv
            smoothed.append(self._vec_to_params(blended, target))
        return smoothed

    def smooth_feature_vectors(
        self,
        prev_indices: List[int],
        effective_indices: List[int],
        blend_lambda: float,
        allocator: PermutationFeatureSlotAllocator,
    ) -> torch.Tensor:
        """CBF: blend previous and target feature embeddings (+ residual-free v1)."""
        device = allocator.feature_emb.weight.device
        target_vecs = allocator.feature_emb(
            torch.tensor(effective_indices, dtype=torch.long, device=device)
        )
        if not prev_indices or len(prev_indices) != self.k:
            return target_vecs

        prev_vecs = allocator.feature_emb(
            torch.tensor(prev_indices, dtype=torch.long, device=device)
        )
        gates = allocator.feature_gates[
            torch.tensor(effective_indices, dtype=torch.long, device=device)
        ]
        target_injection = target_vecs * gates

        prev_gates = allocator.feature_gates[
            torch.tensor(prev_indices, dtype=torch.long, device=device)
        ]
        prev_injection = prev_vecs * prev_gates

        lam = torch.tensor(
            [
                min(1.0, blend_lambda * (0.55 if s == ANCHOR_SLOT else 1.0))
                for s in range(self.k)
            ],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        return (1.0 - lam) * prev_injection + lam * target_injection

    def _bigram_profile(self, token_ids: List[int]) -> Dict[Tuple[int, int], float]:
        if len(token_ids) < 2:
            return {}
        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for i in range(len(token_ids) - 1):
            counts[(token_ids[i], token_ids[i + 1])] += 1
        total = float(sum(counts.values()))
        return {k: v / total for k, v in counts.items()}

    def style_jump_penalty(
        self, prefix_ids: List[int], candidate_ids: List[int]
    ) -> float:
        """KL-like divergence between prefix bigrams and candidate bigrams."""
        p_prof = self._bigram_profile(prefix_ids[-24:])
        c_prof = self._bigram_profile(candidate_ids)
        if not p_prof or not c_prof:
            return 0.0
        keys = set(p_prof) | set(c_prof)
        kl = 0.0
        for key in keys:
            p = p_prof.get(key, 1e-6)
            c = c_prof.get(key, 1e-6)
            kl += p * math.log(p / c)
        return max(0.0, kl)

    def discourse_coherence(
        self, discourse: torch.Tensor, candidate_ids: List[int]
    ) -> float:
        if discourse is None or not candidate_ids:
            return 0.0
        dim = discourse.numel()
        cand = torch.zeros(dim, dtype=torch.float32)
        for i, tid in enumerate(candidate_ids):
            cand[(tid + i * 11) % dim] += 0.09
        denom = discourse.norm() * cand.norm()
        if denom < 1e-8:
            return 0.0
        return float(torch.dot(discourse, cand) / denom)

    def prepare_block(
        self,
        allocator: PermutationFeatureSlotAllocator,
        pooled: torch.Tensor,
        target_indices: List[int],
        prev_acceptance_rate: float,
        prev_accepted_len: int,
        full_rejection: bool,
    ) -> Tuple[List[int], List[Dict[str, float]], float, torch.Tensor]:
        """
        Apply geodesic slot step + SPI/CBF smoothing before drafting.
        Returns (effective_indices, smoothed_params, blend_lambda, smoothed_vectors).
        """
        prev_idx = self.state.prev_feature_indices
        if prev_idx is None:
            effective = list(target_indices)
            target_params = allocator.get_feature_params(target_indices)
            self.state.prev_params = [dict(p) for p in target_params]
            blend_lam = 1.0
        else:
            effective = self.geodesic_feature_indices(
                prev_idx, target_indices, CTSB_MAX_SLOT_SWAPS
            )
            target_params = allocator.get_feature_params(effective)
            dist = self.circuit_distance(prev_idx, effective)
            blend_lam = self.compute_blend_lambda(
                prev_acceptance_rate, prev_accepted_len, dist, full_rejection
            )
            prev_params = self.state.prev_params or [
                self._default_params() for _ in range(self.k)
            ]
            target_params = self.smooth_sampling_params(
                prev_params, target_params, blend_lam
            )

        smoothed_vecs = self.smooth_feature_vectors(
            prev_idx or [], effective, blend_lam, allocator
        )
        self.state.lambda_history.append(blend_lam)
        self.state.effective_feature_history.append(
            [PERSONALITY_FEATURES[i] for i in effective]
        )
        return effective, target_params, blend_lam, smoothed_vecs

    def select_commit_path(
        self,
        proposals: List[List[int]],
        path_rates: List[float],
        feature_indices: List[int],
        fused_proposal: List[int],
        fused_rate: float,
        ema_accept: List[float],
    ) -> Tuple[int, List[int], float, str]:
        """
        TAFK: pick a single stream (or fused) by composite coherence score.
        Returns (path_index, tokens, rate, path_kind).
        """
        candidates: List[Tuple[int, List[int], float, str]] = []
        for j in range(len(proposals)):
            candidates.append((j, proposals[j], path_rates[j], "stream"))
        candidates.append((len(proposals), fused_proposal, fused_rate, "fused"))

        best_idx = 0
        best_score = -1e9
        prev_idx = self.state.prev_feature_indices or []
        prefix_ids = self.state.prev_committed_ids
        discourse = self.state.discourse

        for path_j, tokens, rate, kind in candidates:
            score = CTSB_COHERENCE_ALPHA * rate
            score += CTSB_COHERENCE_GAMMA * self.discourse_coherence(discourse, tokens)
            score -= CTSB_COHERENCE_ETA * self.style_jump_penalty(prefix_ids, tokens)
            if kind == "stream" and path_j < len(feature_indices):
                if feature_indices[path_j] in prev_idx:
                    slot = prev_idx.index(feature_indices[path_j])
                    if slot < len(ema_accept):
                        score += CTSB_COHERENCE_KAPPA * ema_accept[slot]
            if kind == "stream" and path_j == ANCHOR_SLOT:
                score += 0.04
            if score > best_score:
                best_score = score
                best_idx = path_j

        _, tokens, rate, kind = candidates[best_idx]
        return best_idx, tokens, rate, kind

    def post_block_update(
        self,
        effective_indices: List[int],
        smoothed_params: List[Dict[str, float]],
        accepted_ids: List[int],
        generated_ids: List[int],
    ) -> None:
        self.state.prev_feature_indices = list(effective_indices)
        self.state.prev_params = [dict(p) for p in smoothed_params]
        if accepted_ids:
            self.state.prev_committed_ids = list(generated_ids)


# =============================================================================
# TWO-PHASE GROUP THINK MASK BUILDER (educational / documented)
# =============================================================================
@dataclass
class GroupThinkMask:
    """
    In a future kernel integration (custom attention forward inside the denoiser)
    we would build a 4-D mask (or use a custom FlashAttention kernel) with:
        - Phase 1 (draft): block-diagonal + causal prefix. Each of the K agents
          sees its own history + its own block with full (non-causal) attention.
        - Phase 2 (integration): after the K blocks are materialized, we allow
          controlled cross-agent attention (e.g. every agent can attend to the
          last 1-2 "summary" tokens of every other agent) before the final
          verification step.
    Here we only materialize a symbolic description + a tiny tensor for logging.
    """

    k: int
    block_size: int
    phase: str = "draft"  # "draft" or "integration"

    def describe(self) -> str:
        return (
            f"GroupThinkMask(k={self.k}, block={self.block_size}, phase={self.phase}) - "
            "full intra-block + ancestor-causal inter-block (see docstring)"
        )

    def as_tensor(self) -> torch.Tensor:
        """Return a tiny [K, K*block] illustration mask (for debug prints only)."""
        m = self.k * self.block_size
        mask = torch.zeros(self.k, m)
        for agent in range(self.k):
            start = agent * self.block_size
            end = start + self.block_size
            mask[agent, start:end] = 1.0  # full inside own block
            # ancestor prefix (all previous tokens) would be 1s in real mask
        if self.phase == "integration":
            mask.fill_(0.6)  # everybody talks to everybody a little
        return mask


# =============================================================================
# CUMPROD ACCEPTANCE (MBR verification primitive for K parallel trajectories)
# =============================================================================
def compute_accepted_tokens(
    draft_token_ids: List[int],
    target_logprobs_for_exact_tokens: List[float],
    draft_logprobs_for_exact_tokens: List[float],
    min_accept: float = 1e-6,
) -> Tuple[List[int], float]:
    """
    Walk the proposed block left-to-right performing cumprod acceptance vs target logprobs.

    accept_p = min( 1.0 , target_p / max(draft_p, eps) )
    u ~ uniform
    while u < accept_p: keep the token, advance

    Returns (list_of_accepted_ids, acceptance_rate)
    """
    accepted: List[int] = []
    rate = 0.0
    if not draft_token_ids:
        return accepted, rate

    eps = 1e-12
    for t_idx, (tid, lp_t, lp_d) in enumerate(
        zip(
            draft_token_ids,
            target_logprobs_for_exact_tokens,
            draft_logprobs_for_exact_tokens,
        )
    ):
        p_t = math.exp(lp_t) if lp_t is not None else 0.0
        p_d = math.exp(lp_d) if lp_d is not None else 0.15  # neutral if unknown
        p_d = max(p_d, eps)
        accept_p = min(1.0, p_t / p_d)
        u = random.random()
        if u < accept_p:
            accepted.append(tid)
        else:
            # Resample-from-adjusted-distribution step omitted in this orchestration layer.
            # For the benchmark we simply stop the block (conservative, easy to measure).
            break
    rate = len(accepted) / max(1, len(draft_token_ids))
    return accepted, rate


def _logprob_from_entry(entry: Any, token_id: int) -> float:
    """Extract one logprob from a vLLM Logprob object or HF float dict entry."""
    if entry is None:
        return -0.8
    if isinstance(entry, dict):
        if token_id in entry:
            val = entry[token_id]
            if hasattr(val, "logprob"):
                return float(val.logprob)
            return float(val)
        if entry:
            val = next(iter(entry.values()))
            if hasattr(val, "logprob"):
                return float(val.logprob)
            return float(val)
    if hasattr(entry, "token") and int(entry.token) == int(token_id):
        return float(getattr(entry, "logprob", -0.8))
    return -0.8


def extract_logprob_for_token(logprob_dict_or_list: Any, token_id: int) -> float:
    """
    vLLM returns different structures depending on version / prompt_logprobs vs logprobs.
    This helper is defensive (vLLM Logprob objects and HF float dicts).
    """
    try:
        return _logprob_from_entry(logprob_dict_or_list, token_id)
    except Exception:
        pass
    try:
        if isinstance(logprob_dict_or_list, list):
            for entry in logprob_dict_or_list:
                if isinstance(entry, dict) and token_id in entry:
                    return _logprob_from_entry(entry, token_id)
                if hasattr(entry, "token") and int(entry.token) == int(token_id):
                    return float(getattr(entry, "logprob", -0.8))
    except Exception:
        pass
    return -0.9


def extract_target_logprobs_for_draft(
    prompt_logprobs: Optional[List[Optional[Dict[int, Any]]]],
    draft_token_ids: List[int],
) -> List[float]:
    """Map each draft token to its target logprob from the prompt tail."""
    if not prompt_logprobs or not draft_token_ids:
        return [-0.8] * len(draft_token_ids)
    tail_start = len(prompt_logprobs) - len(draft_token_ids)
    out: List[float] = []
    for i, tid in enumerate(draft_token_ids):
        pos = tail_start + i
        if pos < 0 or pos >= len(prompt_logprobs):
            out.append(-0.8)
            continue
        out.append(_logprob_from_entry(prompt_logprobs[pos], tid))
    return out


def make_target_verify_sampling_params() -> SamplingParams:
    """
    Target verification forward: prompt logprobs on the drafted suffix.
    vLLM rejects max_tokens=0; HF fallback ignores the 1 throwaway decode step.
    """
    return SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=True)


# =============================================================================
# DIFFUSIONGEMMA + MILLION-BRAINS CONDITIONED DENOISING
# =============================================================================
def resolve_diffusiongemma_model_path() -> str:
    """
    Resolution order:
      1. KAGGLE_DIFFUSIONGEMMA_DIR (Kaggle Models mount)
      2. Auto-scan /kaggle/input/models/google/diffusiongemma/
      3. HF hub cache
      4. Remote HF model id
    """
    for raw in (LOCAL_DIFFUSIONGEMMA_DIR, KAGGLE_DIFFUSIONGEMMA_DIR):
        if not raw:
            continue
        resolved = resolve_checkpoint_dir(raw)
        if resolved and local_dir_exists(resolved):
            print(f"[DIFFUSION] Kaggle DiffusionGemma checkpoint: {resolved}")
            return resolved

    kaggle_models_root = "/kaggle/input/models/google/diffusiongemma"
    if os.path.isdir(kaggle_models_root):
        found: List[str] = []

        def _walk(base: str, depth: int = 0) -> None:
            if depth > 6:
                return
            if local_dir_exists(base):
                found.append(os.path.abspath(base))
                return
            try:
                for entry in sorted(os.listdir(base)):
                    if entry.startswith("."):
                        continue
                    _walk(os.path.join(base, entry), depth + 1)
            except OSError:
                pass

        _walk(kaggle_models_root)
        if found:
            best = sorted(found, key=lambda p: (-len(p), p))[0]
            print(f"[DIFFUSION] Discovered DiffusionGemma under Kaggle Models: {best}")
            return best

    for candidate in (DIFFUSIONGEMMA_MODEL_PRIMARY, DIFFUSIONGEMMA_MODEL_FALLBACK):
        cache = _hf_hub_snapshot_path(candidate)
        if cache and local_dir_exists(cache):
            print(f"[DIFFUSION] HF cache DiffusionGemma: {cache}")
            return cache
    print(
        f"[DIFFUSION] No local checkpoint — will fetch {DIFFUSIONGEMMA_MODEL_PRIMARY!r} "
        f"(add Kaggle Models input: google/diffusiongemma)"
    )
    return DIFFUSIONGEMMA_MODEL_PRIMARY


def _hf_hub_snapshot_path(model_id: str) -> Optional[str]:
    """Best-effort HuggingFace hub cache lookup without network."""
    try:
        hub = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        slug = model_id.replace("/", "--")
        root = os.path.join(hub, "hub", f"models--{slug}")
        if not os.path.isdir(root):
            return None
        snaps = os.path.join(root, "snapshots")
        if not os.path.isdir(snaps):
            return None
        for rev in sorted(os.listdir(snaps), reverse=True):
            path = os.path.join(snaps, rev)
            if os.path.isfile(os.path.join(path, "config.json")):
                return path
    except Exception:
        pass
    return None


def _arc_engine_max_num_seqs() -> int:
    """Concurrent vLLM sequences needed for batched ARC Phase-1."""
    if not ARC_SLOT_HYPOTHESIS_MODE or not ARC_PHASE1_PROMPT_PARALLELISM:
        return int(DIFFUSION_MAX_NUM_SEQS)
    return max(
        int(DIFFUSION_MAX_NUM_SEQS),
        arc_phase1_slot_batch_size(arc_hypothesis_k()),
    )


def _build_diffusiongemma_vllm_kwargs(
    model_path: str,
    *,
    gpu_memory_utilization: float,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    vLLM-native DiffusionGemma serving config (recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it).
    diffusion_config + entropy_bound sampler + low max_num_seqs to avoid diffusion-state OOM.
    """
    ctx = int(max_model_len or DIFFUSION_MAX_MODEL_LEN)
    seqs = int(max_num_seqs or _arc_engine_max_num_seqs())
    util = min(float(gpu_memory_utilization), float(DIFFUSION_GPU_UTIL))
    kwargs: Dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": "auto",
        "tensor_parallel_size": 1,
        "max_model_len": ctx,
        "max_num_seqs": seqs,
        "max_num_batched_tokens": min(ctx, DIFFUSION_CANVAS_LENGTH * 2),
        "gpu_memory_utilization": util,
        "hf_overrides": {
            "diffusion_sampler": "entropy_bound",
            "diffusion_entropy_bound": float(DIFFUSION_ENTROPY_BOUND),
        },
        "diffusion_config": {"canvas_length": int(DIFFUSION_CANVAS_LENGTH)},
    }
    # vLLM >= gemma image: ignore checkpoint generation_config.json max_tokens cap
    try:
        import inspect

        if "generation_config" in inspect.signature(LLM).parameters:
            kwargs["generation_config"] = "vllm"
    except Exception:
        pass
    return kwargs


def _build_diffusion_conditioned_prompt(
    base_prompt: str,
    feature_name: str,
    feature_params: Dict[str, Any],
    *,
    tokenizer: Any,
    generated_ids: Optional[List[int]] = None,
) -> str:
    """
    Application-layer conditioning injected into each parallel denoising trajectory.
    Surrogate for vLLM ModelState feature embeddings: system bias + self-conditioning canvas.
    """
    lens = SLOT_HYPOTHESIS_LENSES.get(
        feature_name,
        SPATIAL_PRIMITIVE_LENSES.get(
            feature_name, f"Apply the {feature_name} reasoning lens."
        ),
    )
    prior = ""
    if generated_ids:
        prior = tokenizer.decode(generated_ids[-128:], skip_special_tokens=True)
    lines = [
        f"[MBR-DIFFUSION-CONDITION feature={feature_name} "
        f"temp={feature_params.get('temperature', BASE_TEMPERATURE):.2f} "
        f"top_p={feature_params.get('top_p', BASE_TOP_P):.2f}]",
        f"Lens: {lens}",
    ]
    if prior.strip():
        lines.append(f"Prior canvas (self-conditioning): {prior[-400:]}")
    lines.append(base_prompt)
    return "\n".join(lines)


def million_brains_diffusion_denoise_generate(
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
    """
    Hard-wired Million-Brains inside DiffusionGemma denoising.

    Each denoise iteration:
      1) PermutationFeatureSlotAllocator picks K personality features.
      2) CTSB smooths circuit transitions on the pooled canvas state.
      3) K parallel conditioned trajectories propose canvas chunks (feature-biased sampling).
      4) Cross-stream integration (Synthesizer / GroupThinkMask / anchor fusion).
      5) Target verification + cumprod acceptance commits into the live canvas.
      6) Adaptive feature reallocation on under-performing trajectories.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if allocator is None:
        allocator = PermutationFeatureSlotAllocator(
            internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=k
        )
    smoother = (
        CircuitTransitionSmoother(k=k, allocator=allocator)
        if enable_smoothing
        else None
    )

    current_text = prompt
    generated_ids: List[int] = []
    total_accepted = 0
    denoise_steps = 0
    feature_reallocations = 0
    acceptance_history: List[float] = []
    feature_history: List[List[str]] = []
    reframe_events = 0
    prev_acceptance_rate = 0.5
    prev_accepted_len = 0
    ema_accept = [0.5] * k
    active_feature_indices: List[int] = list(range(k))
    max_steps = (max_new_tokens // max(1, block_size)) + 4
    arc_test_input = (arc_context or {}).get("test_input")
    arc_gold = (arc_context or {}).get("gold")
    arc_task_id = str((arc_context or {}).get("task_id") or "")

    if verbose or STREAM_ALL_OUTPUT:
        stream_log(
            f"[MBR-DIFFUSION] start k={k} chunk={block_size} canvas={DIFFUSION_CANVAS_LENGTH} "
            f"max_new={max_new_tokens} smoothing={enable_smoothing}"
        )

    for step in range(max_steps):
        if len(generated_ids) >= max_new_tokens:
            break
        denoise_steps += 1

        pooled = make_pooled_state(generated_ids, step, tokenizer=tokenizer)
        alloc_out = allocator(pooled, step)
        target_feature_indices = alloc_out["feature_indices"]
        target_feature_names = alloc_out["feature_names"]
        feature_history.append(target_feature_names)

        blend_lambda = 1.0
        if smoother is not None:
            full_rejection = prev_accepted_len == 0 and step > 0
            (
                active_feature_indices,
                feature_params,
                blend_lambda,
                _smoothed_vecs,
            ) = smoother.prepare_block(
                allocator,
                pooled,
                target_feature_indices,
                prev_acceptance_rate,
                prev_accepted_len,
                full_rejection,
            )
            active_feature_names = [
                PERSONALITY_FEATURES[i] for i in active_feature_indices
            ]
        else:
            active_feature_indices = target_feature_indices
            active_feature_names = target_feature_names
            feature_params = allocator.get_feature_params(active_feature_indices)

        draft_sampling = []
        conditioned_prompts = []
        for i in range(k):
            p = feature_params[i]
            conditioned_prompts.append(
                _build_diffusion_conditioned_prompt(
                    current_text,
                    active_feature_names[i],
                    p,
                    tokenizer=tokenizer,
                    generated_ids=generated_ids,
                )
            )
            sp = SamplingParams(
                temperature=p["temperature"]
                + (
                    REFRAME_TEMP_BOOST
                    if reframe_events > 0 and step == denoise_steps - 1
                    else 0.0
                ),
                top_p=p["top_p"],
                max_tokens=block_size,
                logprobs=1,
            )
            setattr(
                sp,
                "stream_label",
                f"MBR-denoise/slot{i}/{active_feature_names[i]}",
            )
            draft_sampling.append(sp)

        outs = vllm_llm.generate(conditioned_prompts, draft_sampling, use_tqdm=False)
        proposals: List[List[int]] = []
        draft_lps: List[List[float]] = []
        for slot_i, out in enumerate(outs):
            if not out.outputs:
                proposals.append([])
                draft_lps.append([])
                continue
            completion = out.outputs[0]
            tok_ids = list(completion.token_ids)[:block_size]
            lps = []
            if completion.logprobs:
                for tid in tok_ids:
                    lp = extract_logprob_for_token(
                        completion.logprobs[-len(tok_ids) :], tid
                    )
                    lps.append(lp)
            else:
                lps = [-0.7] * len(tok_ids)
            proposals.append(tok_ids)
            draft_lps.append(lps)
            if verbose or STREAM_ALL_OUTPUT:
                feat_name = active_feature_names[slot_i]
                draft_txt = tokenizer.decode(tok_ids, skip_special_tokens=True)
                stream_log(
                    f"  [DENOISE slot {slot_i}] {feat_name}: {draft_txt!r}"
                )

        fused_proposal: List[int] = []
        if smoother is not None:
            anchor = min(ANCHOR_SLOT, len(proposals) - 1)
            fused_proposal = list(proposals[anchor][:block_size])
        else:
            synth_idx = None
            for idx, name in enumerate(active_feature_names):
                if name == "Synthesizer":
                    synth_idx = idx
                    break
            for pos in range(block_size):
                cands = [pr[pos] for pr in proposals if len(pr) > pos]
                if not cands:
                    break
                if synth_idx is not None and len(proposals[synth_idx]) > pos:
                    fused_proposal.append(proposals[synth_idx][pos])
                else:
                    cnt = Counter(cands)
                    fused_proposal.append(cnt.most_common(1)[0][0])
            if len(fused_proposal) < block_size:
                fused_proposal += proposals[0][len(fused_proposal) : block_size]

        candidates_for_verify = []
        base = current_text
        for pr in proposals:
            candidates_for_verify.append(
                base + tokenizer.decode(pr, skip_special_tokens=True)
            )
        fused_text = tokenizer.decode(fused_proposal, skip_special_tokens=True)
        candidates_for_verify.append(base + fused_text)

        verify_params = make_target_verify_sampling_params()
        setattr(verify_params, "stream_label", f"MBR-denoise-verify/step{step}")
        setattr(verify_params, "verify_only", True)
        setattr(verify_params, "verify_tail_tokens", block_size + 4)
        verify_outs = vllm_llm.generate(
            candidates_for_verify, verify_params, use_tqdm=False
        )

        target_lps_per_path: List[List[float]] = []
        for j, vout in enumerate(verify_outs):
            target_ids = proposals[j] if j < len(proposals) else fused_proposal
            target_lps_per_path.append(
                extract_target_logprobs_for_draft(vout.prompt_logprobs, target_ids)
            )

        best_accepted: List[int] = []
        best_rate = -1.0
        path_rates: List[float] = []
        accepted_per_path: List[List[int]] = []
        for j in range(k):
            acc, rate = compute_accepted_tokens(
                proposals[j], target_lps_per_path[j], draft_lps[j]
            )
            path_rates.append(rate)
            accepted_per_path.append(acc)
            if smoother is None and rate > best_rate:
                best_rate = rate
                best_accepted = acc

        acc_f, rate_f = compute_accepted_tokens(
            fused_proposal, target_lps_per_path[-1], draft_lps[0] if draft_lps else []
        )
        if smoother is not None:
            path_idx, _, _, path_kind = smoother.select_commit_path(
                proposals,
                path_rates,
                active_feature_indices,
                fused_proposal,
                rate_f,
                ema_accept,
            )
            if path_kind == "fused":
                best_accepted = acc_f
                best_rate = rate_f
            else:
                best_accepted = accepted_per_path[path_idx]
                best_rate = path_rates[path_idx]
        elif rate_f > best_rate:
            best_accepted = acc_f
            best_rate = rate_f

        accepted_len = len(best_accepted)
        if accepted_len > 0:
            append_text = tokenizer.decode(best_accepted, skip_special_tokens=True)
            if is_degenerate_arc_token_run(append_text):
                accepted_len = 0
            else:
                generated_ids.extend(best_accepted)
                current_text = current_text + append_text
                total_accepted += accepted_len
                if verbose or STREAM_ALL_OUTPUT:
                    stream_log(
                        f"  [DENOISE-COMMIT step={step:02d} +{accepted_len} tok] "
                        f"{append_text!r}"
                    )

        acceptance_history.append(best_rate)
        prev_acceptance_rate = best_rate
        prev_accepted_len = accepted_len

        if smoother is not None:
            smoother.update_discourse(
                pooled, generated_ids, best_accepted, current_text
            )
            smoother.post_block_update(
                active_feature_indices,
                feature_params,
                best_accepted,
                generated_ids,
            )

        if enable_reallocation:
            for i in range(k):
                ema_accept[i] = 0.7 * ema_accept[i] + 0.3 * path_rates[i]
            for i in range(k):
                if ema_accept[i] < ACCEPTANCE_THRESHOLD:
                    unused = [
                        r
                        for r in range(NUM_PERSONALITY_FEATURES)
                        if r not in active_feature_indices
                    ]
                    if unused:
                        new_feat = unused[(step * 31 + i) % len(unused)]
                        feature_reallocations += 1
                        ema_accept[i] = 0.55

        if accepted_len == 0 and enable_reallocation:
            reframe_events += 1

        if verbose or STREAM_ALL_OUTPUT:
            stream_log(
                f"    Denoise step {step:02d} | features={active_feature_names} | "
                f"accepted={accepted_len}/{block_size} | λ={blend_lambda:.2f}"
                f" | mask=GroupThinkMask(k={k}, block={block_size}, phase=integration)"
            )

        if len(generated_ids) >= max_new_tokens:
            break

    return {
        "generation_mode": "diffusiongemma_mbr_conditioned_denoise",
        "final_text": current_text,
        "generated_ids": generated_ids,
        "num_tokens": len(generated_ids),
        "num_superblocks": denoise_steps,
        "num_denoise_steps": denoise_steps,
        "total_accepted": total_accepted,
        "avg_accepted_per_block": total_accepted / max(1, denoise_steps),
        "feature_reallocations": feature_reallocations,
        "acceptance_history": acceptance_history,
        "feature_history": feature_history,
        "reframe_events": reframe_events,
        "circuit_smoothing_enabled": enable_smoothing,
        "avg_blend_lambda": (
            float(np.mean(smoother.state.lambda_history))
            if smoother and smoother.state.lambda_history
            else 1.0
        ),
        "effective_feature_history": (
            smoother.state.effective_feature_history if smoother else feature_history
        ),
        "diffusion_canvas_length": DIFFUSION_CANVAS_LENGTH,
    }


# =============================================================================
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


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/working")


def network_available(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    """Return False when Kaggle internet is off or DNS cannot resolve the host."""
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        return True
    except OSError:
        return False


def any_download_channel_available() -> bool:
    """True if at least one known model download endpoint resolves."""
    for host in ("kaggle.com", "huggingface.co", "hf.co"):
        if network_available(host):
            return True
    return False


def local_dir_exists(path: str) -> bool:
    """True if a local model dir exists and contains a config.json (i.e. looks like a real HF checkpoint)."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "config.json"))


def resolve_checkpoint_dir(path: str, max_depth: int = 3) -> Optional[str]:
    """
    Return the directory that directly contains config.json.
    Handles Kaggle notebook-input mounts where weights may be one level nested.
    """
    if not path or not os.path.isdir(path):
        return None
    if local_dir_exists(path):
        return os.path.abspath(path)

    found: List[str] = []

    def _walk(base: str, depth: int) -> None:
        if depth > max_depth or not os.path.isdir(base):
            return
        if local_dir_exists(base):
            found.append(os.path.abspath(base))
            return
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            return
        for entry in entries:
            _walk(os.path.join(base, entry), depth + 1)

    _walk(path, 0)
    if not found:
        return None
    return sorted(found, key=_resolve_checkpoint_priority)[0]


def _resolve_checkpoint_priority(path: str) -> Tuple[int, int, int, str]:
    """Prefer full causal-LM checkpoints when multiple mounts exist."""
    name = os.path.basename(path).lower()
    draft_rank = 0
    size_rank = 0
    if "4b" in name or "4-b" in name:
        size_rank -= 10
    elif "3b" in name or "3-b" in name:
        size_rank -= 8
    elif "1.5b" in name or "1_5b" in name:
        size_rank -= 5
    if "qwen" in name:
        size_rank -= 3
    return (draft_rank, size_rank, -len(name), path)


def _checkpoint_priority(path: str) -> Tuple[int, int, str]:
    """Lower tuple sorts earlier: prefer larger models, then lexicographic."""
    name = os.path.basename(path).lower()
    score = 0
    if "qwen" in name:
        score -= 20
    if "1.5b" in name or "1_5b" in name:
        score -= 5
    if "3b" in name or "3-b" in name:
        score -= 8
    if "4b" in name or "4-b" in name:
        score -= 10
    if "7b" in name or "7-b" in name:
        score -= 12
    return (score, -len(name), path)


def discover_hf_hub_cache_checkpoints() -> List[str]:
    """Find materialized snapshots under the HuggingFace hub cache."""
    found: List[str] = []
    for cache_root in (
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
    ):
        if not os.path.isdir(cache_root):
            continue
        try:
            entries = os.listdir(cache_root)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith("models--"):
                continue
            snapshots_dir = os.path.join(cache_root, entry, "snapshots")
            if not os.path.isdir(snapshots_dir):
                continue
            try:
                snaps = os.listdir(snapshots_dir)
            except OSError:
                continue
            for snap in snaps:
                snap_path = os.path.join(snapshots_dir, snap)
                if local_dir_exists(snap_path):
                    found.append(os.path.abspath(snap_path))
    return found


def discover_local_checkpoints(
    extra_roots: Optional[List[str]] = None,
    max_depth: int = 4,
) -> List[str]:
    """
    Scan Kaggle input/working trees (and HF hub cache) for HuggingFace-style checkpoints.
    Typical dataset mount: /kaggle/input/<dataset>/<model-dir>/config.json
    """
    roots: List[str] = []
    if extra_roots:
        roots.extend(extra_roots)
    for root in (
        "/kaggle/working",
        "/kaggle/input",
        os.path.expanduser("~/.cache/kagglehub"),
        "/root/.cache/kagglehub",
    ):
        if os.path.isdir(root) and root not in roots:
            roots.append(root)

    found: List[str] = list(discover_hf_hub_cache_checkpoints())

    def _walk(base: str, depth: int) -> None:
        if depth > max_depth or not os.path.isdir(base):
            return
        if local_dir_exists(base):
            found.append(os.path.abspath(base))
            return
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            return
        for entry in entries:
            _walk(os.path.join(base, entry), depth + 1)

    for root in roots:
        _walk(root, 0)

    ranked = sorted(set(found), key=_checkpoint_priority)
    return ranked


def resolve_local_model_path(prefer_generation: bool = True) -> Optional[str]:
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

def _prefetch_target_dir() -> str:
    short = PREFETCH_MODEL_ID.split("/")[-1]
    if _on_kaggle():
        return os.path.join("/kaggle/working", short)
    return os.path.join(os.getcwd(), short)


def prefetch_via_kagglehub(target_dir: str) -> Optional[str]:
    """Download via Kaggle Models API — works when huggingface.co DNS fails."""
    if not KAGGLEHUB_MODEL_HANDLE:
        return None
    try:
        import kagglehub

        print(
            f"[PREFETCH] Kaggle Models: {KAGGLEHUB_MODEL_HANDLE} -> {target_dir}"
        )
        downloaded = kagglehub.model_download(
            KAGGLEHUB_MODEL_HANDLE,
            output_dir=target_dir,
        )
        for candidate in (downloaded, target_dir):
            if local_dir_exists(candidate):
                print(f"[PREFETCH] Kaggle Models cached at {candidate}")
                return os.path.abspath(candidate)
    except Exception as e:
        print(f"[PREFETCH] Kaggle Models failed ({type(e).__name__}: {e})")
    return None


def prefetch_via_kaggle_dataset() -> Optional[str]:
    """Fallback: attach/download a public Kaggle dataset that ships the weights."""
    if not KAGGLE_DATASET_HANDLE:
        return None
    try:
        import kagglehub

        print(f"[PREFETCH] Kaggle Dataset: {KAGGLE_DATASET_HANDLE}")
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        discovered = discover_local_checkpoints(extra_roots=[dataset_path], max_depth=5)
        if discovered:
            print(f"[PREFETCH] Dataset checkpoint: {discovered[0]}")
            return discovered[0]
    except Exception as e:
        print(f"[PREFETCH] Kaggle Dataset failed ({type(e).__name__}: {e})")
    return None


def prefetch_via_huggingface(target_dir: str) -> Optional[str]:
    """Download from HuggingFace Hub when DNS to huggingface.co/hf.co works."""
    if not network_available("huggingface.co") and not network_available("hf.co"):
        return None
    try:
        from huggingface_hub import snapshot_download

        print(
            f"[PREFETCH] HuggingFace: {PREFETCH_MODEL_ID} -> {target_dir}"
        )
        snapshot_download(
            repo_id=PREFETCH_MODEL_ID,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        if local_dir_exists(target_dir):
            print(f"[PREFETCH] HuggingFace cached at {target_dir}")
            return target_dir
    except Exception as e:
        print(f"[PREFETCH] HuggingFace failed ({type(e).__name__}: {e})")
    return None


def ensure_model_available() -> Optional[str]:
    """
    Ensure a usable on-disk checkpoint exists before vLLM load.
    Order: existing local -> HF hub cache -> Kaggle Models -> Kaggle Dataset -> HuggingFace.
    """
    existing = resolve_local_model_path()
    if existing:
        return existing

    if not AUTO_PREFETCH_TO_WORKING:
        return None

    if not any_download_channel_available():
        print(
            "[PREFETCH] No download channel reachable (Kaggle Internet is likely OFF). "
            "Turn Internet ON in notebook Settings, or attach a model dataset."
        )
        return None

    target_dir = _prefetch_target_dir()
    if local_dir_exists(target_dir):
        print(f"[PREFETCH] Reusing cached model at {target_dir}")
        return target_dir

    for fetcher in (
        lambda: prefetch_via_kagglehub(target_dir),
        prefetch_via_kaggle_dataset,
        lambda: prefetch_via_huggingface(target_dir),
    ):
        path = fetcher()
        if path and local_dir_exists(path):
            return path

    return resolve_local_model_path()


def maybe_prefetch_model_to_working() -> Optional[str]:
    """Backward-compatible alias used by main()."""
    return ensure_model_available()


def pick_model_name() -> Tuple[str, str]:
    """Resolve DiffusionGemma checkpoint."""
    path = resolve_diffusiongemma_model_path()
    return path, "vllm-diffusion"

def _read_hf_config(model_path: str) -> Dict[str, Any]:
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_checkpoint_weight_keys(model_path: str) -> List[str]:
    """Read tensor key names from safetensors index or shard files."""
    keys: List[str] = []
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            keys.extend(list(index_data.get("weight_map", {}).keys()))
        except Exception:
            pass
    if keys:
        return keys

    safe_open = None
    try:
        from safetensors import safe_open as _safe_open

        safe_open = _safe_open
    except ImportError:
        pass

    try:
        entries = sorted(os.listdir(model_path))
    except OSError:
        return keys

    for fname in entries:
        if not fname.endswith(".safetensors"):
            continue
        if "model" not in fname.lower():
            continue
        fpath = os.path.join(model_path, fname)
        if safe_open is not None:
            try:
                with safe_open(fpath, framework="pt") as shard:
                    keys.extend(list(shard.keys()))
            except Exception:
                pass
        else:
            keys.append(fname)
    return keys


def _checkpoint_has_weight_files(model_path: str) -> bool:
    try:
        for fname in os.listdir(model_path):
            if fname.endswith((".safetensors", ".bin")) and "model" in fname.lower():
                return True
    except OSError:
        pass
    return False


def _checkpoint_vocab_size(cfg: Dict[str, Any]) -> Optional[int]:
    vocab = cfg.get("vocab_size")
    if vocab:
        return int(vocab)
    text_cfg = cfg.get("text_config") or {}
    vocab = text_cfg.get("vocab_size")
    return int(vocab) if vocab else None


def _checkpoint_model_info(model_path: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    try:
        cfg = _read_hf_config(model_path)
    except Exception:
        pass
    archs = [str(a) for a in (cfg.get("architectures") or [])]
    model_type = str(cfg.get("model_type", ""))
    return {
        "architectures": archs,
        "model_type": model_type,
        "vocab_size": _checkpoint_vocab_size(cfg),
        "is_qwen35": model_type == "qwen3_5"
        or any("Qwen3_5" in a for a in archs),
        "is_multimodal": bool(cfg.get("vision_config") or cfg.get("video_token_id")),
        "is_text_generator": any(
            tag in a
            for a in archs
            for tag in ("CausalLM", "ConditionalGeneration", "ImageTextToText")
        ),
    }


def bundle_folder_role(model_path: str) -> Optional[str]:
    """Classify checkpoint folder role (DiffusionGemma = base generation)."""
    if not model_path:
        return None
    name = os.path.basename(model_path).lower()
    if "diffusiongemma" in name or "diffusion" in name:
        return "base"
    if is_full_causal_lm_checkpoint(model_path):
        return "base"
    return None


def print_checkpoint_diagnostics(model_path: str, indent: str = "        ") -> None:
    """Emit architecture/weight hints to simplify Kaggle debugging."""
    if not model_path or not local_dir_exists(model_path):
        print(f"{indent}(missing or no config.json)")
        return
    info = _checkpoint_model_info(model_path)
    keys = _list_checkpoint_weight_keys(model_path)
    has_embed = any("embed_tokens" in k for k in keys)
    has_fc = any(".fc.weight" in k or k.endswith("fc.weight") for k in keys)
    print(
        f"{indent}arch={info['architectures']} model_type={info['model_type']!r} "
        f"vocab={info['vocab_size']} embed={has_embed} fc={has_fc} "
        f"weights={len(keys)} shard(s)"
    )


def is_full_causal_lm_checkpoint(model_path: str) -> bool:
    """Positive check: checkpoint has embedding table(s) and can generate text."""
    if not local_dir_exists(model_path):
        return False

    if bundle_folder_role(model_path) == "base":
        return True

    keys = _list_checkpoint_weight_keys(model_path)
    if any("embed_tokens" in k for k in keys):
        return True
    if any("lm_head" in k for k in keys):
        return True

    try:
        cfg = _read_hf_config(model_path)
        info = _checkpoint_model_info(model_path)
        if not info["is_text_generator"]:
            return False
        if not info["vocab_size"]:
            return False
        if _checkpoint_has_weight_files(model_path):
            return True
    except Exception:
        pass
    return False










def resolve_generation_model_path(requested_path: str) -> str:
    """Resolve DiffusionGemma checkpoint path."""
    resolved = resolve_checkpoint_dir(requested_path) or requested_path
    if not local_dir_exists(resolved):
        resolved = resolve_diffusiongemma_model_path()
    print(f"[LOAD] DiffusionGemma checkpoint: {resolved}")
    return resolved

def _guess_causal_lm_architecture(config: Dict[str, Any]) -> Optional[str]:
    model_type = str(config.get("model_type", "")).lower()
    archs = [str(a) for a in (config.get("architectures") or [])]
    if model_type == "qwen3_5" or any("Qwen3_5" in a for a in archs):
        return "Qwen3_5ForConditionalGeneration"
    mapping = {
        "qwen3_moe": "Qwen3MoeForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
        "qwen2_moe": "Qwen2MoeForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        "llama": "LlamaForCausalLM",
        "mistral": "MistralForCausalLM",
    }
    for key, arch in mapping.items():
        if key in model_type:
            return arch
    for arch in archs:
        if "CausalLM" in arch or "ConditionalGeneration" in arch:
            return arch
    return None


def _should_skip_vllm(model_path: str) -> bool:
    """Qwen3.5 multimodal checkpoints routinely fail vLLM on Kaggle; prefer HF."""
    if PREFER_HF_INFERENCE and not ARC_TRY_VLLM:
        return True
    if not SKIP_VLLM_FOR_QWEN35:
        return False
    try:
        info = _checkpoint_model_info(model_path)
        return info["is_qwen35"] or info["is_multimodal"]
    except Exception:
        return False


def _load_hf_generation_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    local_only: bool,
) -> Any:
    """Load the correct HF class for plain LMs and Qwen3.5 multimodal text generation."""
    info = _checkpoint_model_info(model_path)
    common: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": local_only and os.path.isdir(model_path),
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    errors: List[str] = []

    if info["is_qwen35"] or info["is_multimodal"]:
        print(
            f"[LOAD][HF] Qwen3.5/multimodal checkpoint: "
            f"arch={info['architectures']} — using text-only generate path"
        )
        for label, loader in (
            ("AutoModelForImageTextToText", "transformers.AutoModelForImageTextToText"),
            ("AutoModelForCausalLM", "transformers.AutoModelForCausalLM"),
        ):
            try:
                import importlib

                mod = importlib.import_module("transformers")
                cls = getattr(mod, label)
                return cls.from_pretrained(model_path, **common)
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **common)
    except Exception as exc:
        errors.append(f"AutoModelForCausalLM: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "[LOAD][HF] Failed to load generation model from "
            f"{model_path}.\n  " + "\n  ".join(errors)
        ) from exc


def _needs_custom_vllm_handling(model_path: str) -> bool:
    path_l = model_path.lower()
    if any(tag in path_l for tag in ("diffusiongemma", "qwen3", "custom")):
        return True
    try:
        cfg = _read_hf_config(model_path)
    except Exception:
        return False
    archs = [str(a) for a in (cfg.get("architectures") or [])]
    if any("Embedding" in a or "Pooling" in a for a in archs):
        return True
    if cfg.get("is_embedding_model") or cfg.get("pooler_config"):
        return True
    return False


def _cuda_vram_snapshot(device: int = 0) -> Dict[str, float]:
    """Return total/free GiB on a CUDA device (0 if unavailable)."""
    if not torch.cuda.is_available():
        return {"total_gib": 0.0, "free_gib": 0.0, "allocated_gib": 0.0}
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
    except Exception:
        total_b = torch.cuda.get_device_properties(device).total_memory
        free_b = total_b - torch.cuda.memory_allocated(device)
    return {
        "total_gib": total_b / (1024**3),
        "free_gib": free_b / (1024**3),
        "allocated_gib": (total_b - free_b) / (1024**3),
    }


def _adaptive_vllm_gpu_util(requested: float) -> float:
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

def _release_cuda_cache() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _build_vllm_attempts(
    model_path: str,
    gpu_memory_utilization: float,
) -> List[Dict[str, Any]]:
    kwargs = _build_diffusiongemma_vllm_kwargs(
        model_path, gpu_memory_utilization=gpu_memory_utilization
    )
    return [kwargs]


@dataclass
class _HFCompletionOutput:
    token_ids: List[int]
    text: str
    logprobs: Optional[List[Dict[int, float]]] = None


@dataclass
class _HFRequestOutput:
    outputs: List[_HFCompletionOutput]
    prompt_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


_STREAM_IN_THINKING = False


def stream_begin(label: str) -> None:
    """Open a labeled live-generation block."""
    if STREAM_GENERATION:
        print(f"\n[STREAM] ▶ {label}", flush=True)


def stream_end(label: str = "") -> None:
    """Close a live-generation block."""
    if STREAM_GENERATION:
        suffix = f" — {label}" if label else ""
        print(f"\n[STREAM] ■ end{suffix}", flush=True)


def stream_emit(text: str, *, end: str = "") -> None:
    """Print one decoded token chunk immediately (thinking tags included)."""
    global _STREAM_IN_THINKING
    if not STREAM_GENERATION or not text:
        return
    if STREAM_PRINT_THINKING:
        lower = text.lower()
        if "<think>" in lower and not _STREAM_IN_THINKING:
            _STREAM_IN_THINKING = True
            print("\n[THINKING] ", end="", flush=True)
        print(text, end=end, flush=True)
        if "</think>" in lower:
            _STREAM_IN_THINKING = False
            print("\n[ANSWER] ", end="", flush=True)
        return
    print(text, end=end, flush=True)


def stream_log(message: str) -> None:
    """Print a non-token diagnostic line during streaming."""
    if STREAM_GENERATION or STREAM_ALL_OUTPUT:
        print(message, flush=True)


def _sampling_params_label(sp: Any, fallback: str) -> str:
    return str(getattr(sp, "stream_label", None) or fallback)


def _top_p_sample_logits(logits: torch.Tensor, temperature: float, top_p: float) -> Tuple[int, float]:
    """Sample one token ID + logprob from logits."""
    temp = max(float(temperature), 1e-5)
    probs = torch.softmax(logits / temp, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > float(top_p)
        if mask.any():
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
        next_local = torch.multinomial(sorted_probs, num_samples=1)
        next_id = int(sorted_idx[next_local].item())
        logp = float(torch.log(probs[next_id] + 1e-12).item())
        return next_id, logp
    next_id = int(torch.multinomial(probs, num_samples=1).item())
    logp = float(torch.log(probs[next_id] + 1e-12).item())
    return next_id, logp


class HFGenerateEngine:
    """Fallback engine when vLLM cannot load DiffusionGemma."""

    def __init__(
        self,
        model_path: str,
        tokenizer: Any,
        *,
        local_only: bool = True,
    ):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[LOAD][HF] Loading causal LM from {model_path} ...")
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.model = _load_hf_generation_model(
            model_path,
            dtype=dtype,
            local_only=local_only,
        ).eval()
        self.device = next(self.model.parameters()).device
        emb = self.model.get_input_embeddings()
        if emb is None or emb.weight.numel() == 0:
            raise RuntimeError(
                f"[LOAD][HF] Model at {model_path} has no usable embed_tokens."
            )
        print("[LOAD][HF] HuggingFace generate engine ready (vLLM-compatible API).")

    def _encode(self, text: str) -> torch.Tensor:
        prompt = text if text and str(text).strip() else " "
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max(4096, arc_vllm_context_budget()),
        )
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] == 0:
            fallback_id = (
                getattr(self.tokenizer, "bos_token_id", None)
                or self.tokenizer.eos_token_id
                or 0
            )
            input_ids = torch.tensor([[fallback_id]], device=self.device, dtype=torch.long)
        return input_ids

    def _decode_ids(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _prompt_logprobs_for_ids(
        self,
        input_ids: torch.Tensor,
        *,
        tail_count: Optional[int] = None,
    ) -> List[Optional[Dict[int, float]]]:
        """Target logprobs for prompt tokens. tail_count limits work to draft suffix."""
        seq_len = int(input_ids.shape[1])
        if seq_len < 2:
            return [None]

        with torch.inference_mode():
            logits = self.model(input_ids).logits[0]

        if tail_count is not None and HF_FAST_VERIFY:
            tail_count = max(1, min(int(tail_count), seq_len - 1))
            start_pos = seq_len - tail_count
            log_probs = torch.log_softmax(logits[start_pos - 1 : seq_len - 1], dim=-1)
            token_ids = input_ids[0, start_pos:seq_len]
            gathered = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            out: List[Optional[Dict[int, float]]] = [None] * start_pos
            for i, tid in enumerate(token_ids.tolist()):
                out.append({tid: float(gathered[i].item())})
            return out

        token_ids = input_ids[0, 1:seq_len]
        log_probs = torch.log_softmax(logits[: seq_len - 1], dim=-1)
        gathered = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        out = [None]
        for i, tid in enumerate(token_ids.tolist()):
            out.append({tid: float(gathered[i].item())})
        return out

    def _sample_ids_streaming(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        *,
        stream_label: str,
    ) -> Tuple[List[int], List[Dict[int, float]]]:
        """Token-by-token generation with live stdout streaming."""
        global _STREAM_IN_THINKING
        _STREAM_IN_THINKING = False
        stream_begin(stream_label)

        eos_ids = {
            int(t)
            for t in (
                self.tokenizer.eos_token_id,
                getattr(self.tokenizer, "eod_id", None),
            )
            if t is not None
        }

        new_ids: List[int] = []
        step_logprobs: List[Dict[int, float]] = []
        past: Any = None
        cur = input_ids

        for _step in range(int(max_tokens)):
            with torch.inference_mode():
                if past is None:
                    outputs = self.model(cur, use_cache=True)
                else:
                    outputs = self.model(
                        cur[:, -1:],
                        past_key_values=past,
                        use_cache=True,
                    )
            past = outputs.past_key_values
            logits = outputs.logits[0, -1, :]

            if temperature <= 0:
                next_id = int(logits.argmax().item())
                logp = float(
                    torch.log_softmax(logits, dim=-1)[next_id].item()
                )
            else:
                next_id, logp = _top_p_sample_logits(
                    logits, temperature=temperature, top_p=top_p
                )

            new_ids.append(next_id)
            step_logprobs.append({next_id: logp})

            piece = self.tokenizer.decode([next_id], skip_special_tokens=False)
            stream_emit(piece)

            if next_id in eos_ids:
                break

            next_tensor = torch.tensor(
                [[next_id]], device=cur.device, dtype=cur.dtype
            )
            cur = torch.cat([cur, next_tensor], dim=1)

        stream_end(stream_label)
        return new_ids, step_logprobs

    def _sample_ids(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        *,
        stream_label: str = "GENERATE",
        stream: bool = True,
    ) -> Tuple[List[int], List[Dict[int, float]]]:
        if input_ids.shape[1] == 0:
            raise ValueError("Refusing to generate from empty input_ids")

        if STREAM_GENERATION and stream and not ARC_FAST_INFERENCE:
            return self._sample_ids_streaming(
                input_ids,
                max_tokens,
                temperature,
                top_p,
                stream_label=stream_label,
            )

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": pad_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(float(temperature), 1e-5)
            gen_kwargs["top_p"] = float(top_p)

        with torch.inference_mode():
            outputs = self.model.generate(input_ids, **gen_kwargs)

        prompt_len = input_ids.shape[1]
        new_ids = outputs.sequences[0, prompt_len:].tolist()
        step_logprobs: List[Dict[int, float]] = []
        if outputs.scores:
            for step_idx, tid in enumerate(new_ids):
                if step_idx >= len(outputs.scores):
                    break
                logp = torch.log_softmax(outputs.scores[step_idx][0], dim=-1)[tid].item()
                step_logprobs.append({tid: logp})
        else:
            step_logprobs = [{tid: 0.0} for tid in new_ids]
        return new_ids, step_logprobs

    def generate(
        self, prompts: List[str], sampling_params: Any
    ) -> List[_HFRequestOutput]:
        """
        vLLM-compatible generate. Accepts one SamplingParams for all prompts,
        or a list of SamplingParams (one per prompt) for heterogeneous K-stream drafting.
        """
        results: List[_HFRequestOutput] = []
        if isinstance(sampling_params, list):
            sp_list: List[Any] = list(sampling_params)
            if not sp_list:
                sp_list = [SamplingParams(max_tokens=16)]
            if len(sp_list) < len(prompts):
                sp_list.extend([sp_list[-1]] * (len(prompts) - len(sp_list)))
        else:
            sp_list = [sampling_params] * len(prompts)

        for batch_idx, (prompt, sp) in enumerate(zip(prompts, sp_list)):
            max_tokens = int(getattr(sp, "max_tokens", 0) or 0)
            temperature = float(getattr(sp, "temperature", 1.0) or 0.0)
            top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
            want_prompt_logprobs = bool(getattr(sp, "prompt_logprobs", False))
            verify_only = bool(getattr(sp, "verify_only", False))
            verify_tail = getattr(sp, "verify_tail_tokens", None)
            stream_label = _sampling_params_label(
                sp, f"GENERATE[{batch_idx}]"
            )

            # HF verify: logprobs only — skip throwaway 1-token decode (vLLM uses max_tokens=1).
            if want_prompt_logprobs and max_tokens <= 1:
                max_tokens = 0

            input_ids = self._encode(prompt)
            if (
                STREAM_ALL_OUTPUT
                and not ARC_FAST_INFERENCE
                and batch_idx == 0
                and not verify_only
            ):
                preview = prompt[-240:] if len(prompt) > 240 else prompt
                stream_log(
                    f"[HF] prompt tail ({len(prompt)} chars): ...{preview}"
                )
            elif verify_only and STREAM_ALL_OUTPUT and not ARC_FAST_INFERENCE and batch_idx == 0:
                stream_log(
                    f"[HF] verify-only pass (tail_logprobs={verify_tail or 'full'}, "
                    f"n_candidates={len(prompts)})"
                )
            prompt_logprobs = None
            if want_prompt_logprobs:
                tail_n = int(verify_tail) if verify_tail is not None else None
                prompt_logprobs = self._prompt_logprobs_for_ids(
                    input_ids, tail_count=tail_n
                )
            if max_tokens <= 0:
                results.append(_HFRequestOutput(outputs=[], prompt_logprobs=prompt_logprobs))
                continue

            do_stream = STREAM_GENERATION and not (
                verify_only or (not STREAM_VERIFY_PASSES and want_prompt_logprobs)
            )
            new_ids, step_logprobs = self._sample_ids(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream_label=stream_label,
                stream=do_stream,
            )
            results.append(
                _HFRequestOutput(
                    outputs=[
                        _HFCompletionOutput(
                            token_ids=new_ids,
                            text=self._decode_ids(new_ids),
                            logprobs=step_logprobs,
                        )
                    ],
                    prompt_logprobs=prompt_logprobs,
                )
            )
        return results


def create_inference_engine(
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

def verify_inference_engine(llm: Any, tokenizer: Any) -> None:
    """Fail fast before ARC eval if the engine cannot tokenize/generate."""
    engine_label = (
        "HF-fallback"
        if isinstance(llm, HFGenerateEngine)
        else getattr(type(llm), "__name__", str(type(llm)))
    )
    gen_path = getattr(llm, "model_path", None)
    print("\n[VERIFY] Inference engine smoke test (parent/coordinator engine)")
    print(f"    script   : {SCRIPT_VERSION}")
    print(f"    engine   : {engine_label}")
    print(
        f"    arc_eval : Phase1={arc_hypothesis_k()} proposals/engine, "
        f"Phase2=1 grid/engine (see ARC PIPELINE banner at eval start)"
    )
    if torch.cuda.is_available():
        snap = _cuda_vram_snapshot()
        print(
            f"    cuda     : {torch.cuda.device_count()} device(s), "
            f"cuda:0 free {snap['free_gib']:.2f}/{snap['total_gib']:.2f} GiB"
        )
    if gen_path:
        print(f"    generate : {gen_path}")
    ctx = get_inference_max_context(llm)
    print(f"    max_ctx  : {ctx} tokens (vLLM max_model_len / HF cap)")
    if isinstance(llm, HFGenerateEngine) and ARC_TRY_VLLM:
        print(
            "    [WARN] HF fallback active — ARC eval will be ~10-50x slower than vLLM. "
            "Fix vLLM load or set ARC_TRY_VLLM=False if intentional."
        )
    if ctx < arc_vllm_context_budget():
        print(
            f"    [WARN] Engine context {ctx} < ARC budget {arc_vllm_context_budget()} — "
            "30x30 tasks may fail. Restart kernel and let vLLM load with max_model_len>=8192."
        )

    probe = "ARC smoke test."
    encoded = tokenizer(probe, return_tensors="pt", add_special_tokens=True)
    if encoded["input_ids"].shape[1] == 0:
        raise RuntimeError("[VERIFY] Tokenizer produced empty input_ids.")

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    setattr(sp, "stream_label", "VERIFY-smoke")
    try:
        out = llm.generate([probe], sp)[0]
    except Exception as exc:
        hint = ""
        if "cannot reshape tensor of 0 elements" in str(exc):
            hint = (
                f"\n  Likely cause: stale script (need SCRIPT_VERSION={SCRIPT_VERSION} in banner)."
            )
        raise RuntimeError(
            f"[VERIFY] Engine smoke test failed: {type(exc).__name__}: {exc}{hint}"
        ) from exc

    n_new = len(out.outputs[0].token_ids) if out.outputs else 0
    if n_new < 1:
        raise RuntimeError(
            "[VERIFY] Engine returned zero new tokens on smoke prompt."
        )
    print(f"    smoke    : OK ({n_new} token generated)")

    bench_prompt = "Decode throughput probe. Count to twenty: "
    bench_tokens = 48
    bench_sp = SamplingParams(temperature=0.0, max_tokens=bench_tokens)
    setattr(bench_sp, "stream_label", "VERIFY-bench")
    t0 = time.perf_counter()
    try:
        bench_out = llm.generate([bench_prompt], bench_sp)[0]
        bench_elapsed = time.perf_counter() - t0
        bench_n = len(bench_out.outputs[0].token_ids) if bench_out.outputs else 0
        bench_tps = bench_n / max(bench_elapsed, 1e-6)
        print(
            f"    bench    : {bench_n} tok in {bench_elapsed:.2f}s "
            f"({bench_tps:.1f} tok/s decode, short prompt)"
        )
    except Exception as exc:
        print(f"    bench    : skipped ({type(exc).__name__}: {exc})")

    if not isinstance(llm, HFGenerateEngine):
        long_tokens = 64
        long_prompt = "ARC final-pass decode probe. " + ("context " * 800)
        long_sp = SamplingParams(temperature=0.0, max_tokens=long_tokens)
        setattr(long_sp, "stream_label", "VERIFY-long-bench")
        t1 = time.perf_counter()
        try:
            long_out = llm.generate([long_prompt], long_sp)[0]
            long_elapsed = time.perf_counter() - t1
            long_n = len(long_out.outputs[0].token_ids) if long_out.outputs else 0
            long_tps = long_n / max(long_elapsed, 1e-6)
            long_in = count_prompt_tokens(tokenizer, long_prompt)
            print(
                f"    long_bench: {long_n} tok in {long_elapsed:.2f}s "
                f"({long_tps:.1f} tok/s, prompt~{long_in}tok)"
            )
        except Exception as exc:
            print(f"    long_bench: skipped ({type(exc).__name__}: {exc})")


def load_models(model_name: str):
    """Load DiffusionGemma inference engine + tokenizer."""
    local_only = os.path.isdir(model_name) and local_dir_exists(model_name)
    gen_path = resolve_generation_model_path(model_name) if local_only else model_name
    load_label = gen_path if local_only else f"{gen_path} (remote)"
    print(
        f"\n[LOAD] Initializing DiffusionGemma engine for {load_label}"
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

def load_local_models() -> Tuple[LLM, Any, Optional[LLM], Optional[Any]]:
    """Load local DiffusionGemma without network access."""
    print("\n[LOCAL-LOAD] DiffusionGemma paths:")
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
            "[LOCAL-LOAD] No DiffusionGemma checkpoint found.\n"
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

def discover_kaggle_arc_competition_dir() -> Optional[str]:
    """Find the ARC Prize competition mount under /kaggle/input/competitions."""
    if os.path.isdir(KAGGLE_ARC_COMPETITION_DIR):
        return KAGGLE_ARC_COMPETITION_DIR
    comp_root = "/kaggle/input/competitions"
    if not os.path.isdir(comp_root):
        return None
    try:
        entries = sorted(os.listdir(comp_root))
    except OSError:
        return None
    for name in entries:
        if "arc" not in name.lower():
            continue
        candidate = os.path.join(comp_root, name)
        if os.path.isfile(
            os.path.join(candidate, "arc-agi_training_challenges.json")
        ) or os.path.isfile(
            os.path.join(candidate, "arc-agi_evaluation_challenges.json")
        ):
            return candidate
    return None


def arc_paths_for_split(base_dir: str, split: str) -> Tuple[str, str]:
    if split not in ARC_SPLIT_FILES:
        raise ValueError(
            f"Unknown ARC_DATA_SPLIT={split!r}. Use one of: {list(ARC_SPLIT_FILES)}"
        )
    challenges_name, solutions_name = ARC_SPLIT_FILES[split]
    return (
        os.path.join(base_dir, challenges_name),
        os.path.join(base_dir, solutions_name),
    )


def arc_split_files_present(base_dir: str, split: str) -> bool:
    challenges_path, solutions_path = arc_paths_for_split(base_dir, split)
    return os.path.isfile(challenges_path) and os.path.isfile(solutions_path)


def resolve_arc_eval_paths(
    *,
    profile: Optional[str] = None,
    split: Optional[str] = None,
    challenges_override: Optional[str] = None,
    solutions_override: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Resolve ARC challenges/solutions paths from profile toggles.

    Returns (challenges_path, solutions_path, source_label).
    """
    profile = profile or ARC_DATA_PROFILE
    split = split or ARC_DATA_SPLIT

    if challenges_override and solutions_override:
        return challenges_override, solutions_override, "explicit override"

    if profile == "off":
        return None, None, "off"

    if challenges_override or solutions_override:
        return challenges_override, solutions_override, "partial override"

    if profile in ("kaggle", "auto"):
        kaggle_dir = discover_kaggle_arc_competition_dir()
        if kaggle_dir and arc_split_files_present(kaggle_dir, split):
            ch, sol = arc_paths_for_split(kaggle_dir, split)
            return ch, sol, f"kaggle:{kaggle_dir}"
        if profile == "kaggle":
            ch, sol = arc_paths_for_split(KAGGLE_ARC_COMPETITION_DIR, split)
            return ch, sol, f"kaggle:{KAGGLE_ARC_COMPETITION_DIR}"

    if profile in ("local", "auto"):
        local_dir = os.path.abspath(LOCAL_ARC_DATA_DIR)
        if arc_split_files_present(local_dir, split):
            ch, sol = arc_paths_for_split(local_dir, split)
            return ch, sol, f"local:{local_dir}"

    return None, None, "unresolved"


def print_arc_data_config(
    challenges_path: Optional[str],
    solutions_path: Optional[str],
    source: str,
) -> None:
    print("\n[ARC DATA]")
    print(f"    profile : {ARC_DATA_PROFILE}")
    print(f"    split   : {ARC_DATA_SPLIT}")
    print(f"    source  : {source}")
    if challenges_path:
        print(f"    challenges: {challenges_path}")
    if solutions_path:
        print(f"    solutions : {solutions_path}")
    if ARC_DATA_PROFILE == "local":
        print(
            "    Kaggle paste tip: set ARC_DATA_PROFILE = \"kaggle\" at the top of this file."
        )
    if not challenges_path or not solutions_path:
        print("    ARC eval: disabled (demo benchmark only).")


def _require_dataset_file(path: Optional[str], label: str) -> str:
    if not path:
        raise ValueError(
            f"{label} path is required for ARC evaluation. "
            "Pass --eval-challenges and --eval-solutions, or set EVAL_*_PATH toggles."
        )
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_arc_dataset(challenges_path: str, solutions_path: str) -> Dict[str, Any]:
    """Load paired ARC-AGI challenges + solutions JSON files."""
    challenges_path = _require_dataset_file(challenges_path, "Challenges")
    solutions_path = _require_dataset_file(solutions_path, "Solutions")
    with open(challenges_path, "r", encoding="utf-8") as f:
        challenges = json.load(f)
    with open(solutions_path, "r", encoding="utf-8") as f:
        solutions = json.load(f)
    if not isinstance(challenges, dict) or not isinstance(solutions, dict):
        raise ValueError("ARC JSON files must be objects keyed by task id.")
    return {"challenges": challenges, "solutions": solutions}


def _format_grid_ascii_compact(grid: List[List[int]]) -> str:
    """Row-per-line digits (0-9) for readable ARC grids in prompts."""
    if not grid:
        return "(empty)"
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _format_grid_minified_json(grid: List[List[int]]) -> str:
    """Single-line JSON — fewer tokens than multiline ASCII for large grids."""
    if not grid:
        return "[]"
    return json.dumps(grid, separators=(",", ":"))


def _estimate_grid_json_tokens(grid: List[List[int]]) -> int:
    """Rough token count for a minified JSON 2D grid."""
    if not grid or not grid[0]:
        return 64
    rows = len(grid)
    cols = len(grid[0])
    cells = rows * cols
    return max(64, int(cells * 2.8 + rows * 12 + 96))


def arc_final_grid_max_tokens(task: Dict[str, Any]) -> int:
    """
    Final-pass output budget: scale with largest train output so 29x30 grids fit.
    Stays within ARC_MBR_OUTPUT_TOKEN_BUDGET (hyp slots share the remainder).
    """
    budget = int(ARC_MBR_OUTPUT_TOKEN_BUDGET)
    floor = max(int(ARC_FINAL_GRID_MIN_TOKENS), budget // 4)
    ceiling = max(floor, int(budget * ARC_FINAL_GRID_MAX_FRACTION))
    need = floor
    for ex in task.get("train", []):
        out = ex.get("output") or []
        if out:
            need = max(need, _estimate_grid_json_tokens(out))
    return min(ceiling, max(floor, need))


def arc_spatial_slot_max_tokens(task: Dict[str, Any], test_index: int = 0) -> int:
    """Per-slot output budget for spatial Phase-1 (one JSON grid, not full final-pass budget)."""
    est = 256
    for ex in task.get("train", []):
        out = ex.get("output") or []
        if out:
            est = max(est, _estimate_grid_json_tokens(out))
    tests = task.get("test") or []
    if test_index < len(tests):
        inp = (tests[test_index].get("input") or [])
        if inp and not any(ex.get("output") for ex in task.get("train", [])):
            est = max(est, _estimate_grid_json_tokens(inp))
    est = min(est + 128, 2048)
    return min(est, arc_hypothesis_max_tokens(task))


def _arc_task_with_train_limit(task: Dict[str, Any], n_train: int) -> Dict[str, Any]:
    train = list(task.get("train") or [])
    n_keep = max(0, min(int(n_train), len(train)))
    return {**task, "train": train[:n_keep]}


def _arc_grid_cell_count(grid: List[List[int]]) -> int:
    if not grid or not grid[0]:
        return 0
    return len(grid) * len(grid[0])


def _arc_largest_grid_cells(task: Dict[str, Any], test_index: int = 0) -> int:
    largest = 0
    for ex in task.get("train") or []:
        for key in ("input", "output"):
            largest = max(largest, _arc_grid_cell_count(ex.get(key) or []))
    tests = task.get("test") or []
    if test_index < len(tests):
        largest = max(
            largest, _arc_grid_cell_count(tests[test_index].get("input") or [])
        )
    return largest


def _arc_task_needs_compact_prompt(
    task: Dict[str, Any], test_index: int = 0, *, cell_threshold: int = 400
) -> bool:
    """True for 20x20+ grids — skip verbose encodings and train-pair sweeps."""
    return _arc_largest_grid_cells(task, test_index) >= int(cell_threshold)


def arc_phase1_slot_batch_size(k: int) -> int:
    """How many Phase-1 slots to pass per vLLM generate() call."""
    if not ARC_PHASE1_PROMPT_PARALLELISM:
        return 1
    return min(max(1, int(k)), max(1, int(ARC_VLLM_MAX_NUM_SEQS)))


def _arc_phase1_generate_slots(
    vllm_llm: Any,
    prompts: List[str],
    sp_list: List[Any],
    *,
    label: str = "phase1",
) -> List[Any]:
    """Run Phase-1 slot generation; serial (batch=1) or batched per ARC_PHASE1_PROMPT_PARALLELISM."""
    k = len(prompts)
    slot_batch = arc_phase1_slot_batch_size(k)
    outs: List[Any] = []
    for start in range(0, k, slot_batch):
        end = min(start + slot_batch, k)
        arc_eval_log(f"{label} generate slots {start + 1}-{end}/{k}")
        outs.extend(
            _vllm_generate_arc(
                vllm_llm,
                prompts[start:end],
                sp_list[start:end],
            )
        )
    return outs




def format_arc_task_body_minimal(
    task_id: str,
    task: Dict[str, Any],
    test_index: int = 0,
) -> str:
    """Test input only — last resort when full train context exceeds vLLM ctx."""
    tests = task.get("test") or []
    if test_index >= len(tests):
        raise IndexError(f"test_index {test_index} out of range for {task_id}")
    test_inp = tests[test_index]["input"]
    n_train = len(task.get("train") or [])
    sh = f"{len(test_inp)}x{len(test_inp[0]) if test_inp else 0}"
    return (
        f"ARC {task_id} ({n_train} train pairs omitted). "
        f"Test input {sh}: {_format_grid_minified_json(test_inp)}"
    )


def arc_hypothesis_k() -> int:
    """Parallel hypothesis proposals per MBR engine (independent of benchmark K=4)."""
    return max(1, min(int(ARC_HYPOTHESIS_SLOTS), NUM_PERSONALITY_FEATURES))


def arc_hypothesis_max_tokens(task: Dict[str, Any]) -> int:
    """Per-slot hypothesis cap — never steal more than half the output budget."""
    budget = int(ARC_MBR_OUTPUT_TOKEN_BUDGET)
    final_reserve = max(
        arc_final_grid_max_tokens(task),
        int(budget * ARC_FINAL_GRID_MIN_FRACTION),
    )
    hyp_pool = max(0, budget - final_reserve)
    per_slot = max(64, hyp_pool // max(1, arc_hypothesis_k()))
    if ARC_HYPOTHESIS_ENABLE_THINKING and not ARC_DISABLE_THINKING:
        think_cap = int(ARC_HYPOTHESIS_THINKING_TOKEN_CAP)
        if think_cap > 0:
            per_slot = min(per_slot, think_cap)
    return per_slot


def arc_mbr_final_output_budget(
    task: Dict[str, Any], hyp_tokens_used: int
) -> int:
    """
    Output tokens for final grid pass = full output budget minus what slots already emitted.
    Unused hypothesis budget flows to the final pass.
    """
    budget = int(ARC_MBR_OUTPUT_TOKEN_BUDGET)
    used = max(0, int(hyp_tokens_used))
    remaining = max(0, budget - used)
    need = arc_final_grid_max_tokens(task)
    ceiling = max(need, int(budget * ARC_FINAL_GRID_MAX_FRACTION))
    return max(int(ARC_FINAL_GRID_MIN_TOKENS), min(ceiling, remaining))


def arc_vllm_context_budget() -> int:
    """Minimum vLLM max_model_len: ARC grids + generation + chat-template slack."""
    raw = (
        int(ARC_MAX_PROMPT_TOKENS)
        + int(ARC_MBR_OUTPUT_TOKEN_BUDGET)
        + int(ARC_CHAT_TEMPLATE_SLACK)
    )
    return int(math.ceil(raw / 256.0) * 256)


def get_inference_max_context(llm: Any) -> int:
    """Effective max context tokens for the loaded engine (vLLM max_model_len or HF cap)."""
    stored = getattr(llm, "max_model_len", None)
    if stored is not None:
        return int(stored)
    try:
        engine = getattr(llm, "llm_engine", None) or getattr(
            getattr(llm, "llm", None), "llm_engine", None
        )
        if engine is not None:
            cfg = getattr(engine, "model_config", None)
            if cfg is not None and hasattr(cfg, "max_model_len"):
                return int(cfg.max_model_len)
    except Exception:
        pass
    return max(4096, int(ARC_MAX_PROMPT_TOKENS))


def format_arc_task_body(
    task_id: str,
    task: Dict[str, Any],
    test_index: int = 0,
    *,
    grid_format: str = "auto",
) -> str:
    """Train + test input context shared by hypothesis slots and final grid pass."""
    if grid_format == "auto":
        grid_format = "minified" if ARC_FAST_INFERENCE else "ascii"

    def _fmt_grid(grid: List[List[int]]) -> str:
        if grid_format in ("minified", "terse"):
            return _format_grid_minified_json(grid)
        if grid_format == "rows":
            return ";".join(" ".join(str(c) for c in row) for row in grid)
        return _format_grid_ascii_compact(grid)

    if grid_format == "rows":
        lines = [f"ARC {task_id} colors0-9 rows=semicolon cols=space"]
        for i, example in enumerate(task.get("train", []), start=1):
            inp, out = example["input"], example["output"]
            in_shape = f"{len(inp)}x{len(inp[0]) if inp else 0}"
            out_shape = f"{len(out)}x{len(out[0]) if out else 0}"
            lines.append(f"T{i}i{in_shape}:{_fmt_grid(inp)}")
            lines.append(f"T{i}o{out_shape}:{_fmt_grid(out)}")
        test_inputs = task.get("test", [])
        if test_index >= len(test_inputs):
            raise IndexError(
                f"Task {task_id} has {len(test_inputs)} test inputs; "
                f"requested index {test_index}."
            )
        test_inp = test_inputs[test_index]["input"]
        test_shape = f"{len(test_inp)}x{len(test_inp[0]) if test_inp else 0}"
        lines.append(f"Xi{test_shape}:{_fmt_grid(test_inp)}")
        return "\n".join(lines)

    if grid_format == "terse":
        lines = [f"ARC {task_id} colors0-9"]
        for i, example in enumerate(task.get("train", []), start=1):
            inp, out = example["input"], example["output"]
            in_shape = f"{len(inp)}x{len(inp[0]) if inp else 0}"
            out_shape = f"{len(out)}x{len(out[0]) if out else 0}"
            lines.append(f"T{i}i{in_shape}:{_fmt_grid(inp)}")
            lines.append(f"T{i}o{out_shape}:{_fmt_grid(out)}")
        test_inputs = task.get("test", [])
        if test_index >= len(test_inputs):
            raise IndexError(
                f"Task {task_id} has {len(test_inputs)} test inputs; "
                f"requested index {test_index}."
            )
        test_inp = test_inputs[test_index]["input"]
        test_shape = f"{len(test_inp)}x{len(test_inp[0]) if test_inp else 0}"
        lines.append(f"Xi{test_shape}:{_fmt_grid(test_inp)}")
        return "\n".join(lines)

    lines = [
        f"ARC-AGI task {task_id}. Infer the grid transformation from the training pairs.",
        "Cell colors are integers 0-9. Output shape may differ from input shape.",
    ]
    for i, example in enumerate(task.get("train", []), start=1):
        inp, out = example["input"], example["output"]
        in_shape = f"{len(inp)}x{len(inp[0]) if inp else 0}"
        out_shape = f"{len(out)}x{len(out[0]) if out else 0}"
        if ARC_FAST_INFERENCE:
            lines.append(f"Train {i} input ({in_shape}): {_fmt_grid(inp)}")
            lines.append(f"Train {i} output ({out_shape}): {_fmt_grid(out)}")
        else:
            lines.append(f"Train {i} input ({in_shape}):")
            lines.append(_format_grid_ascii_compact(inp))
            lines.append(f"Train {i} input JSON: {json.dumps(inp)}")
            lines.append(f"Train {i} output ({out_shape}):")
            lines.append(_format_grid_ascii_compact(out))
            lines.append(f"Train {i} output JSON: {json.dumps(out)}")
    test_inputs = task.get("test", [])
    if test_index >= len(test_inputs):
        raise IndexError(
            f"Task {task_id} has {len(test_inputs)} test inputs; requested index {test_index}."
        )
    test_inp = test_inputs[test_index]["input"]
    test_shape = f"{len(test_inp)}x{len(test_inp[0]) if test_inp else 0}"
    if ARC_FAST_INFERENCE:
        lines.append(f"Test input ({test_shape}): {_fmt_grid(test_inp)}")
    else:
        lines.append(f"Test input ({test_shape}):")
        lines.append(_format_grid_ascii_compact(test_inp))
        lines.append(f"Test input JSON: {json.dumps(test_inp)}")
    return "\n".join(lines)


def resolve_arc_task_body(
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int = 0,
    *,
    max_prompt_tokens: Optional[int] = None,
) -> Tuple[str, int, str]:
    """
    Pick the most detailed grid encoding that fits max_prompt_tokens.
    Returns (body_text, token_count, format_label).
    """
    formats = (
        ("terse", "minified", "ascii")
        if ARC_FAST_INFERENCE
        else ("ascii", "minified", "terse")
    )
    best_body = ""
    best_tok = 0
    best_fmt = "minified"
    for fmt in formats:
        body = format_arc_task_body(
            task_id, task, test_index=test_index, grid_format=fmt
        )
        n_tok = count_prompt_tokens(tokenizer, body)
        best_body, best_tok, best_fmt = body, n_tok, fmt
        if max_prompt_tokens is None or n_tok <= max_prompt_tokens:
            return body, n_tok, fmt
    minimal = format_arc_task_body_minimal(task_id, task, test_index)
    n_min = count_prompt_tokens(tokenizer, minimal)
    if max_prompt_tokens is None or n_min <= max_prompt_tokens:
        return minimal, n_min, "minimal"
    if n_min < best_tok:
        return minimal, n_min, "minimal"
    return best_body, best_tok, best_fmt


def resolve_arc_spatial_task_body(
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int,
    primitive_name: str,
    *,
    system_content: str,
    engine_ctx: int,
    slot_max_tokens: int,
    enable_thinking: bool,
) -> Tuple[str, str, int, str]:
    """
    Pick task-body encoding so the full spatial-slot chat prompt fits
    engine_ctx minus slot output budget (body-only caps miss ~1-2k template overhead).
    """
    max_input = max(256, int(engine_ctx) - int(slot_max_tokens) - 32)
    systems = [system_content]
    if ARC_FAST_INFERENCE:
        systems.append(
            "ARC spatial solver. Output one JSON 2D int array (0-9). No prose."
        )
    if _arc_task_needs_compact_prompt(task, test_index):
        formats: Tuple[str, ...] = ("rows",)
    elif ARC_FAST_INFERENCE:
        formats = ("rows", "terse", "minified", "ascii")
    else:
        formats = ("ascii", "minified", "terse", "rows")
    best: Tuple[str, str, int, str] = ("", system_content, 10**9, "minified")

    for sys_msg in systems:
        for fmt in formats:
            body = format_arc_task_body(
                task_id, task, test_index=test_index, grid_format=fmt
            )
            user_content = build_spatial_grid_user_content(
                task_id,
                task,
                test_index,
                primitive_name,
                task_body=body,
                compact=ARC_FAST_INFERENCE,
            )
            prompt = _wrap_arc_chat_prompt(
                tokenizer,
                user_content,
                system_content=sys_msg,
                enable_thinking=enable_thinking,
                assistant_prefill=ARC_ASSISTANT_PREFILL,
            )
            n_tok = count_prompt_tokens(tokenizer, prompt)
            if n_tok < best[2]:
                best = (body, sys_msg, n_tok, fmt)
            if n_tok <= max_input:
                return body, sys_msg, n_tok, fmt

    for sys_msg in systems:
        minimal = format_arc_task_body_minimal(task_id, task, test_index)
        user_content = build_spatial_grid_user_content(
            task_id,
            task,
            test_index,
            primitive_name,
            task_body=minimal,
            compact=ARC_FAST_INFERENCE,
        )
        prompt = _wrap_arc_chat_prompt(
            tokenizer,
            user_content,
            system_content=sys_msg,
            enable_thinking=enable_thinking,
            assistant_prefill=ARC_ASSISTANT_PREFILL,
        )
        n_tok = count_prompt_tokens(tokenizer, prompt)
        if n_tok < best[2]:
            best = (minimal, sys_msg, n_tok, "minimal")
        if n_tok <= max_input:
            return minimal, sys_msg, n_tok, "minimal"

    return best[0], best[1], best[2], best[3]


def _wrap_arc_chat_prompt(
    tokenizer: Any,
    user_content: str,
    *,
    system_content: str,
    enable_thinking: Optional[bool] = None,
    assistant_prefill: str = "",
) -> str:
    if ARC_USE_CHAT_TEMPLATE:
        return apply_arc_chat_prompt_custom(
            tokenizer,
            user_content,
            system_content=system_content,
            enable_thinking=enable_thinking,
            assistant_prefill=assistant_prefill,
        )
    return user_content


def resolve_arc_hypothesis_bundle(
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int,
    feature_name: str,
    *,
    system_content: str,
    engine_ctx: int,
    max_output_tokens: int,
    enable_thinking: bool,
) -> Tuple[str, str, int, str]:
    """
    Pick task-body encoding + system prompt so the full chat-wrapped hypothesis
    prompt fits engine_ctx. Returns (task_body, system_used, prompt_tokens, fmt).
    """
    max_input = engine_ctx - int(max_output_tokens) - 32
    formats = (
        ("terse", "minified", "ascii")
        if ARC_FAST_INFERENCE
        else ("ascii", "minified", "terse")
    )
    systems = [system_content]
    if ARC_FAST_INFERENCE:
        systems.append(
            "ARC analyst. Describe transformation rules in plain text only — no grids."
        )

    best: Tuple[str, str, int, str] = ("", system_content, 10**9, "minified")
    for sys_msg in systems:
        for fmt in formats:
            body = format_arc_task_body(
                task_id, task, test_index=test_index, grid_format=fmt
            )
            user_content = build_slot_hypothesis_user_content(
                task_id,
                task,
                test_index,
                feature_name,
                task_body=body,
                compact=ARC_FAST_INFERENCE,
            )
            prompt = _wrap_arc_chat_prompt(
                tokenizer,
                user_content,
                system_content=sys_msg,
                enable_thinking=enable_thinking,
            )
            n_tok = count_prompt_tokens(tokenizer, prompt)
            if n_tok < best[2]:
                best = (body, sys_msg, n_tok, fmt)
            if n_tok <= max_input:
                return body, sys_msg, n_tok, fmt
    return best


def resolve_arc_final_prompt_bundle(
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int,
    hypotheses: List[Dict[str, Any]],
    *,
    system_content: str,
    engine_ctx: int,
    final_output_tokens: int,
    enable_thinking: bool = False,
    assistant_prefill: str = ARC_ASSISTANT_PREFILL,
) -> Tuple[str, int, int, str]:
    """
    Build final grid prompt so input+output fits engine_ctx while preserving
    final_output_tokens from the MBR output budget.
    Shrinks hypothesis text and task-body encoding before touching output budget.
    """
    max_input = max(256, engine_ctx - int(final_output_tokens) - 32)
    formats = (
        ("terse", "minified", "ascii")
        if ARC_FAST_INFERENCE
        else ("minified", "terse", "ascii")
    )
    systems = [system_content]
    if ARC_FAST_INFERENCE and system_content not in systems:
        systems.append(
            "ARC solver. Synthesize hypotheses, output one JSON 2D int array (0-9) only."
        )

    best: Tuple[str, int, int, str] = ("", 10**9, 0, "terse")
    for sys_msg in systems:
        for hyp_cap in ARC_FINAL_HYP_CHAR_CAPS:
            for fmt in formats:
                body = format_arc_task_body(
                    task_id, task, test_index=test_index, grid_format=fmt
                )
                final_user = build_final_grid_user_content(
                    task_id,
                    task,
                    test_index,
                    hypotheses,
                    task_body=body,
                    hyp_char_cap=int(hyp_cap),
                )
                prompt = _wrap_arc_chat_prompt(
                    tokenizer,
                    final_user,
                    system_content=sys_msg,
                    enable_thinking=enable_thinking,
                    assistant_prefill=assistant_prefill,
                )
                n_in = count_prompt_tokens(tokenizer, prompt)
                body_tok = count_prompt_tokens(tokenizer, body)
                if n_in < best[1]:
                    best = (prompt, n_in, body_tok, fmt)
                if n_in <= max_input:
                    return prompt, n_in, body_tok, fmt

    prompt, n_in, body_tok, fmt = best
    if n_in > max_input:
        stream_log(
            f"[MBR-WARN] final prompt {n_in}tok > input cap {max_input}tok "
            f"(engine={engine_ctx}, output_reserved={final_output_tokens}). "
            "Using tightest encoding; raise VLLM_MAX_MODEL_LEN if generation truncates."
        )
    return prompt, n_in, body_tok, fmt


def format_arc_prompt(task_id: str, task: Dict[str, Any], test_index: int = 0) -> str:
    """Turn one ARC task into a text prompt with train demos + one test input."""
    lines = [format_arc_task_body(task_id, task, test_index=test_index)]
    if ARC_STRUCTURED_THINKING and STREAM_PRINT_THINKING:
        lines.extend(
            [
                "Inside <think>, reason in numbered steps. After each step emit exactly:",
                "HYPOTHESIS_GRID: [[...]]  (partial or complete output grid; use best guess for unknown cells).",
                "Update HYPOTHESIS_GRID every step as your rule hypothesis improves.",
                "After </think>, output exactly one final JSON 2D array (no markdown, no extra text).",
            ]
        )
    else:
        lines.extend(
            [
                "Output format: a single JSON 2D array of integers (cell colors 0-9 only).",
                "Example: [[0,1,2],[3,4,5]]",
                "Do not include markdown fences, explanations, or any text before/after the array.",
            ]
        )
    lines.append("Test output JSON array:")
    return "\n".join(lines)


def arc_resolve_enable_thinking() -> Optional[bool]:
    """Step 2: explicit control of Qwen3.5 thinking via chat_template enable_thinking."""
    if ARC_DISABLE_THINKING or not ARC_FORCE_ENABLE_THINKING:
        return False
    if STREAM_PRINT_THINKING:
        return True
    return False


def _arc_json_grid_schema() -> Dict[str, Any]:
    """JSON schema for guided decoding: 2D int grid (0-9), up to 30x30."""
    return {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 9},
            "minItems": 1,
            "maxItems": 30,
        },
        "minItems": 1,
        "maxItems": 30,
    }


def _arc_guided_json_enabled() -> bool:
    return bool(ARC_GUIDED_JSON_DECODING)


def _apply_arc_guided_decoding(sp: Any) -> Any:
    """Step 3: attach vLLM guided JSON decoding when available (Outlines/xgrammar)."""
    if not _arc_guided_json_enabled():
        return sp
    try:
        from vllm.sampling_params import GuidedDecodingParams

        schema = _arc_json_grid_schema()
        guided = None
        for factory in (
            lambda: GuidedDecodingParams(json=schema),
            lambda: GuidedDecodingParams(json_object=schema),
            lambda: GuidedDecodingParams(json_schema=schema),
        ):
            try:
                guided = factory()
                break
            except TypeError:
                continue
        if guided is not None:
            sp.guided_decoding = guided
            setattr(sp, "arc_guided_json", True)
    except Exception as exc:
        if ARC_EVAL_VERBOSE:
            stream_log(f"[ARC-GUIDED] guided decoding unavailable ({exc}); greedy parse fallback")
    return sp


def _vllm_generate_arc(
    vllm_llm: Any,
    prompts: List[str],
    sp_list: List[Any],
) -> List[Any]:
    """Stock vLLM generate with thinking disabled and optional guided JSON."""
    gen_kwargs: Dict[str, Any] = {}
    if arc_resolve_enable_thinking() is False:
        gen_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        return vllm_llm.generate(
            prompts, sp_list, use_tqdm=False, **gen_kwargs
        )
    except TypeError:
        try:
            return vllm_llm.generate(
                prompts,
                sp_list,
                use_tqdm=False,
                tokenization_kwargs=gen_kwargs.get("chat_template_kwargs"),
            )
        except TypeError:
            return vllm_llm.generate(prompts, sp_list, use_tqdm=False)


def apply_arc_chat_prompt_custom(
    tokenizer: Any,
    user_content: str,
    *,
    system_content: str,
    enable_thinking: Optional[bool] = None,
    assistant_prefill: str = "",
) -> str:
    """Chat-wrap with explicit system/thinking/prefill (hypothesis vs final-grid passes)."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    template_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = bool(enable_thinking)
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
        template_kwargs["add_generation_prompt"] = False
        template_kwargs["continue_final_message"] = True

    if not hasattr(tokenizer, "apply_chat_template"):
        return user_content

    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as exc:
            print(f"[ARC] chat template failed ({exc}); using raw prompt.")
    except Exception as exc:
        print(f"[ARC] chat template failed ({exc}); using raw prompt.")
    return user_content


def apply_arc_chat_prompt(tokenizer: Any, user_content: str) -> str:
    """Wrap ARC content with the model's chat template (required for Qwen3.5)."""
    if STREAM_PRINT_THINKING:
        if ARC_STRUCTURED_THINKING:
            system_content = (
                "You solve ARC-AGI grid puzzles. Inside <think>, use numbered steps. "
                "After each step write HYPOTHESIS_GRID: followed by a JSON 2D array of "
                "integers 0-9 (partial guesses allowed). Refine HYPOTHESIS_GRID each step. "
                "Example inside think:\n"
                "Step 1: Rows with color 8 form bands.\n"
                "HYPOTHESIS_GRID: [[8,8],[8,0]]\n"
                "Step 2: Fill band interiors with color 4.\n"
                "HYPOTHESIS_GRID: [[8,8],[4,4]]\n"
                "After </think> output exactly one final JSON 2D array. No markdown."
            )
        else:
            system_content = (
                "You solve ARC-AGI grid puzzles. Think step-by-step inside <think>...</think> "
                "tags, then reply with exactly one JSON 2D array of integers (values 0-9). "
                "No markdown fences."
            )
    else:
        system_content = (
            "You solve ARC-AGI grid puzzles. Reply with exactly one JSON 2D array "
            "of integers (values 0-9). No markdown, no explanation."
        )
    enable_thinking: Optional[bool] = arc_resolve_enable_thinking()
    prefill = ARC_ASSISTANT_PREFILL if enable_thinking is not True else ""
    return apply_arc_chat_prompt_custom(
        tokenizer,
        user_content,
        system_content=system_content,
        enable_thinking=enable_thinking,
        assistant_prefill=prefill,
    )


def build_arc_inference_prompt(
    tokenizer: Any, task_id: str, task: Dict[str, Any], test_index: int = 0
) -> str:
    raw = format_arc_prompt(task_id, task, test_index=test_index)
    if ARC_USE_CHAT_TEMPLATE:
        return apply_arc_chat_prompt(tokenizer, raw)
    return raw


def _strip_model_artifacts(text: str) -> str:
    """Remove thinking blocks and markdown fences from model output."""
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _repair_truncated_json_array(fragment: str) -> Optional[str]:
    """Close unbalanced brackets and drop a dangling partial final row."""
    s = fragment.strip()
    if not s.startswith("["):
        return None
    s = re.sub(r",\s*\[[^\]]*$", "", s)
    s = re.sub(r",\s*$", "", s)
    opens = s.count("[") - s.count("]")
    if opens > 0:
        s = s + ("]" * opens)
    return s


def _balanced_bracket_span_at(text: str, start: int) -> Optional[str]:
    """Return one balanced [...] span from start, or a repaired truncated span."""
    if start < 0 or start >= len(text) or text[start] != "[":
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
            if depth == 0:
                return text[start : j + 1]
    return _repair_truncated_json_array(text[start:])


def _balanced_bracket_spans(text: str) -> List[str]:
    """Extract all bracket spans with proper nesting."""
    spans: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        span = _balanced_bracket_span_at(text, i)
        if span:
            spans.append(span)
            if text[i : i + len(span)] == span:
                i += len(span)
            else:
                i += 1
        else:
            i += 1
    return spans


def _grid_attempt_text_variants(text: str) -> List[str]:
    """
    Qwen often emits a broken first grid then restarts with a second [[...]].
    Try the full text, the suffix from the last [[, and prefill repairs.
    """
    variants: List[str] = []
    if not text:
        return variants

    def _add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)

    _add(text)
    last_anchor = text.rfind("[[")
    if last_anchor > 0:
        _add(text[last_anchor:])
    for anchor in re.finditer(r"\[\[", text):
        _add(text[anchor.start() :])

    stripped = text.lstrip()
    if stripped and not stripped.startswith("["):
        _add((ARC_ASSISTANT_PREFILL or "[[") + stripped)
    elif (
        ARC_ASSISTANT_PREFILL
        and stripped.startswith("[")
        and not stripped.startswith(ARC_ASSISTANT_PREFILL)
    ):
        _add(ARC_ASSISTANT_PREFILL + stripped[1:])
    return variants


def _is_valid_arc_grid(parsed: Any) -> bool:
    """True only for non-empty rectangular 2D grids with ARC colors 0-9."""
    if not isinstance(parsed, list) or not parsed:
        return False
    if not all(isinstance(row, list) and row for row in parsed):
        return False
    width = len(parsed[0])
    if not all(len(row) == width for row in parsed):
        return False
    return all(
        isinstance(cell, int) and 0 <= cell <= 9
        for row in parsed
        for cell in row
    )


def _parse_grid_from_row_pattern(text: str) -> Optional[List[List[int]]]:
    """Fallback: stitch [n,n,n] row literals after the last [[ restart."""
    anchor = text.rfind("[[")
    chunk = text[anchor:] if anchor >= 0 else text
    rows: List[List[int]] = []
    width: Optional[int] = None
    for match in re.finditer(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", chunk):
        try:
            row = json.loads("[" + match.group(1) + "]")
        except json.JSONDecodeError:
            rows = []
            width = None
            continue
        if not row or not all(isinstance(c, int) and 0 <= c <= 9 for c in row):
            rows = []
            width = None
            continue
        if width is None:
            width = len(row)
        if len(row) != width:
            rows = []
            width = None
            continue
        rows.append(row)
    return rows if rows else None


HYPOTHESIS_GRID_MARKER = "HYPOTHESIS_GRID"


def _extract_thinking_content(text: str) -> str:
    """Return content inside the last <think> block (handles unclosed blocks)."""
    if not text:
        return ""
    lower = text.lower()
    start_tag = "<think>"
    end_tag = "</think>"
    start = lower.rfind(start_tag)
    if start < 0:
        return ""
    start += len(start_tag)
    end = lower.find(end_tag, start)
    if end < 0:
        return text[start:]
    return text[start:end]


def _parse_grid_from_fragment(fragment: str) -> Tuple[Optional[List[List[int]]], str]:
    """Parse the best valid or repaired grid from a text fragment."""
    if not fragment:
        return None, ""
    seen_raw: set = set()
    best: Optional[Tuple[int, List[List[int]], str]] = None
    for variant in _grid_attempt_text_variants(fragment):
        for anchor in re.finditer(r"\[\[", variant):
            raw = _balanced_bracket_span_at(variant, anchor.start())
            if not raw or raw in seen_raw:
                continue
            seen_raw.add(raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if _is_valid_arc_grid(parsed):
                best = (anchor.start(), parsed, raw)
        for raw in _balanced_bracket_spans(variant):
            if raw in seen_raw:
                continue
            seen_raw.add(raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if _is_valid_arc_grid(parsed):
                pos = variant.rfind(raw)
                if best is None or pos >= best[0]:
                    best = (pos, parsed, raw)
        row_grid = _parse_grid_from_row_pattern(variant)
        if row_grid is not None:
            return row_grid, json.dumps(row_grid, separators=(",", ":"))
    if best is not None:
        return best[1], best[2]
    repaired = _repair_truncated_json_array(fragment[fragment.rfind("[[") :] if "[[" in fragment else fragment)
    if repaired:
        try:
            parsed = json.loads(repaired)
            if _is_valid_arc_grid(parsed):
                return parsed, repaired
        except json.JSONDecodeError:
            pass
    return None, fragment.strip()[:120]


def parse_hypothesis_grid_from_thinking(
    text: str,
) -> Tuple[Optional[List[List[int]]], str, int]:
    """
    Extract the latest partial output grid from structured thinking.
    Returns (grid_or_none, raw_fragment, step_count).
    """
    thinking = _extract_thinking_content(text)
    search_space = thinking if thinking else text
    markers = list(
        re.finditer(r"HYPOTHESIS_GRID\s*:", search_space, flags=re.IGNORECASE)
    )
    step_count = len(markers)
    if markers:
        fragment = search_space[markers[-1].end() :]
        grid, raw = _parse_grid_from_fragment(fragment)
        if grid is not None:
            return grid, raw, step_count
    if thinking:
        grid, raw = _parse_grid_from_fragment(thinking)
        if grid is not None:
            return grid, raw, max(step_count, 1)
    return None, search_space.strip()[-120:], step_count


def is_degenerate_arc_token_run(decoded_text: str) -> bool:
    """True when a decoded super-block adds no puzzle content (think-tag spam)."""
    if not decoded_text or not decoded_text.strip():
        return True
    cleaned = decoded_text.strip()
    stripped = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    stripped = stripped.replace("<|im_end|>", "").strip()
    if not stripped:
        return True
    if re.fullmatch(r"[\s,]+", stripped):
        return True
    if re.fullmatch(r"</think>", stripped, flags=re.IGNORECASE):
        return True
    if ARC_STRUCTURED_THINKING and HYPOTHESIS_GRID_MARKER.lower() not in stripped.lower():
        if not re.search(r"\d", stripped) and "[" not in stripped:
            return True
    return False


def _grid_shape_label(grid: Optional[List[List[int]]]) -> str:
    if not grid:
        return "—"
    h = len(grid)
    w = len(grid[0]) if grid and grid[0] else 0
    return f"{h}x{w}"


def _assistant_suffix(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :]
    return full_text


def print_mbr_slot_inference_state(
    *,
    sb: int,
    phase: str,
    slot_i: Optional[int],
    feature_name: str,
    test_input: Optional[List[List[int]]],
    gold: Optional[List[List[int]]],
    assistant_suffix: str,
    task_id: str = "",
) -> None:
    """Print full ASCII matrices for test input, gold, and current hypothesis."""
    if not ARC_PRINT_STEP_MATRICES:
        return
    hyp_grid, hyp_raw, step_n = parse_hypothesis_grid_from_thinking(assistant_suffix)
    slot_part = f"slot={slot_i} " if slot_i is not None else ""
    header = (
        f"[MBR-MATRIX] sb={sb:02d} phase={phase} {slot_part}"
        f"{feature_name}"
    )
    if task_id:
        header += f" task={task_id}"
    if step_n:
        header += f" hypothesis_step={step_n}"
    stream_log(header)

    blocks: List[List[str]] = []
    if test_input:
        blocks.append(
            _render_grid_ascii(
                test_input,
                title=f"TEST INPUT ({_grid_shape_label(test_input)})",
            )
        )
    if gold:
        blocks.append(
            _render_grid_ascii(gold, title=f"GOLD ({_grid_shape_label(gold)})")
        )
    if hyp_grid is not None:
        diff_title = f"HYPOTHESIS ({_grid_shape_label(hyp_grid)})"
        if gold:
            blocks.append(
                _render_grid_ascii(
                    hyp_grid,
                    title=diff_title,
                    diff_against=gold,
                    pred_for_diff=hyp_grid,
                )
            )
        else:
            blocks.append(_render_grid_ascii(hyp_grid, title=diff_title))
    else:
        blocks.append(
            [
                "HYPOTHESIS (unparsed)",
                f"  tail: {hyp_raw!r}",
            ]
        )
    for line in _align_grid_columns(blocks):
        stream_log("  " + line)


def parse_arc_answer_grid(text: str) -> Optional[List[List[int]]]:
    """Prefer final grid after </think>; else latest HYPOTHESIS_GRID in thinking."""
    cleaned = _strip_model_artifacts(text or "")
    post_think = cleaned
    if "</think>" in (text or "").lower():
        post_think = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
        post_think = _strip_model_artifacts(post_think)
    final = parse_grid_from_text(post_think, only_after=None)
    if final is not None:
        return final
    hyp, _, _ = parse_hypothesis_grid_from_thinking(text or "")
    return hyp


def extract_arc_generated_suffix(result: Dict[str, Any], prompt: str) -> str:
    """Return only model-new text (exclude the prompt prefix)."""
    if result.get("generated_text"):
        gen = str(result["generated_text"])
    else:
        final = str(result.get("final_text", ""))
        if final.startswith(prompt):
            gen = final[len(prompt) :]
        else:
            gen = final
    keep_thinking = ARC_FINAL_ENABLE_THINKING and not ARC_DISABLE_THINKING
    if not keep_thinking:
        gen = _strip_model_artifacts(gen)
    thinking_in_gen = "<think>" in (gen or "").lower() or "</think>" in (gen or "").lower()
    if (
        ARC_ASSISTANT_PREFILL
        and gen
        and not gen.lstrip().startswith("[")
        and not thinking_in_gen
        and not keep_thinking
    ):
        gen = ARC_ASSISTANT_PREFILL + gen
    return gen


def _extract_transformation_hypothesis(text: str) -> str:
    """Pull TRANSFORMATION_HYPOTHESIS block or return cleaned prose."""
    if not text:
        return ""
    search_spaces: List[str] = [text]
    thinking = _extract_thinking_content(text)
    if thinking:
        search_spaces.append(thinking)
    post_think = text
    if "</think>" in text.lower():
        post_think = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    search_spaces.extend([post_think, _strip_model_artifacts(text)])

    seen: set = set()
    for search in search_spaces:
        if not search or search in seen:
            continue
        seen.add(search)
        match = re.search(
            r"TRANSFORMATION_HYPOTHESIS\s*:\s*([\s\S]+)",
            search,
            flags=re.IGNORECASE,
        )
        if match:
            body = match.group(1).strip()
            body = re.split(
                r"\n(?:TRANSFORMATION_HYPOTHESIS|HYPOTHESIS_GRID)\s*:", body, flags=re.I
            )[0]
            return body.strip()
    return _strip_model_artifacts(text).strip()


def build_spatial_grid_user_content(
    task_id: str,
    task: Dict[str, Any],
    test_index: int,
    primitive_name: str,
    *,
    task_body: Optional[str] = None,
    compact: bool = False,
) -> str:
    """User message for one spatial primitive slot: emit a JSON grid hypothesis."""
    lens = SPATIAL_PRIMITIVE_LENSES.get(
        primitive_name,
        "Infer the spatial transformation from train pairs and apply it to the test input.",
    )
    body = task_body or format_arc_task_body(task_id, task, test_index=test_index)
    if compact:
        return "\n".join(
            [
                body,
                f"Primitive:{primitive_name}",
                f"Lens:{lens}",
                "Output one JSON 2D int array (0-9) for the test output grid.",
            ]
        )
    return "\n".join(
        [
            body,
            "",
            f"Spatial primitive: {primitive_name}",
            f"Lens: {lens}",
            "",
            "Using ONLY this geometric lens, predict the test output grid.",
            "Reply with exactly one JSON 2D array of integers (colors 0-9).",
            "No markdown, no explanation, no thinking tags — JSON array only.",
            "Test output JSON array:",
        ]
    )


def build_slot_hypothesis_user_content(
    task_id: str,
    task: Dict[str, Any],
    test_index: int,
    feature_name: str,
    *,
    task_body: Optional[str] = None,
    compact: bool = False,
) -> str:
    """User message for one feature-slot: analyze patterns, never output a grid."""
    lens = SLOT_HYPOTHESIS_LENSES.get(
        feature_name,
        "Describe the transformation rule that maps each train input to its output.",
    )
    body = task_body or format_arc_task_body(task_id, task, test_index=test_index)
    if compact:
        return "\n".join(
            [
                body,
                f"Role:{feature_name}",
                f"Lens:{lens}",
                "TRANSFORMATION_HYPOTHESIS:",
                "(text rule, <120 words, no grid)",
            ]
        )
    return "\n".join(
        [
            body,
            "",
            f"Feature-slot role: {feature_name}",
            f"Your lens: {lens}",
            "",
            "Do NOT output any JSON grid or digit matrix.",
            "Reply with a concise textual analysis using this exact header:",
            "TRANSFORMATION_HYPOTHESIS:",
            "then your rule/pattern/algorithm in plain text (concise; under 120 words).",
        ]
    )


def build_final_grid_user_content(
    task_id: str,
    task: Dict[str, Any],
    test_index: int,
    hypotheses: List[Dict[str, Any]],
    *,
    task_body: Optional[str] = None,
    hyp_char_cap: int = 600,
) -> str:
    """User message for final pass: synthesize slot hypotheses into one output grid."""
    body = task_body or format_arc_task_body(task_id, task, test_index=test_index)
    hyp_lines = []
    for h in hypotheses:
        text = str(h.get("text", ""))
        if hyp_char_cap <= 0:
            text = "(see slot analysis)"
        elif len(text) > hyp_char_cap:
            text = text[: max(0, hyp_char_cap - 3)] + "..."
        hyp_lines.append(f"  [{h['slot']}] {h['feature']}: {text}")
    return "\n".join(
        [
            body,
            "",
            "Parallel transformation hypotheses from feature-slots (synthesize, do not copy blindly):",
            *hyp_lines,
            "",
            "Using the train pairs and the hypotheses above, produce the test output grid.",
            "Output exactly one JSON 2D array of integers (colors 0-9).",
            "No markdown, no explanation before or after the array.",
            "Test output JSON array:",
        ]
    )


def collect_feature_slot_hypotheses(
    vllm_llm: Any,
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int,
    *,
    k: Optional[int] = None,
    allocator: Optional[PermutationFeatureSlotAllocator] = None,
    seed: int = 42,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """K parallel one-shot passes: each slot proposes a textual transformation rule."""
    k = arc_hypothesis_k() if k is None else int(k)
    random.seed(seed)
    if allocator is None:
        allocator = PermutationFeatureSlotAllocator(
            internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=k
        )
    pooled = make_pooled_state([], 0, tokenizer=tokenizer)
    alloc_out = allocator(pooled, 0)
    feature_indices = alloc_out["feature_indices"]
    feature_names = alloc_out["feature_names"]
    feature_params = allocator.get_feature_params(feature_indices)

    hyp_thinking = (
        ARC_HYPOTHESIS_ENABLE_THINKING
        and arc_resolve_enable_thinking() is True
    )
    if hyp_thinking:
        system_content = (
            "You are an ARC-AGI pattern analyst. Study grid transformations from train pairs. "
            "Reason inside </think> tags. Never output a JSON grid or digit matrix — only "
            "describe rules, patterns, and algorithms in plain text after thinking."
        )
    else:
        system_content = (
            "You are an ARC-AGI pattern analyst. Study grid transformations from train pairs. "
            "Never output a JSON grid or digit matrix in this phase — only describe rules, "
            "patterns, and algorithms in plain text."
        )

    hypotheses: List[Dict[str, Any]] = []
    engine_ctx = get_inference_max_context(vllm_llm)
    hyp_max_tokens = arc_hypothesis_max_tokens(task)
    probe_feat = max(
        feature_names,
        key=lambda n: len(SLOT_HYPOTHESIS_LENSES.get(n, n)),
    )
    task_body, system_used, probe_tok, body_fmt = resolve_arc_hypothesis_bundle(
        tokenizer,
        task_id,
        task,
        test_index,
        probe_feat,
        system_content=system_content,
        engine_ctx=engine_ctx,
        max_output_tokens=int(hyp_max_tokens),
        enable_thinking=hyp_thinking,
    )
    prompts: List[str] = []
    sp_list: List[Any] = []
    slot_meta: List[Tuple[int, str, Dict[str, Any]]] = []
    max_needed = 0
    total_prompt_tok = 0

    if verbose or (STREAM_ALL_OUTPUT and not ARC_FAST_INFERENCE):
        stream_log(
            f"[MBR-HYP] batched {k} slot hypotheses "
            f"(max_tokens={hyp_max_tokens}/slot, thinking={hyp_thinking}) "
            f"engine_ctx={engine_ctx} prompt~{probe_tok}tok({body_fmt}) "
            f"features={feature_names}"
        )

    for slot_i in range(k):
        feat_name = feature_names[slot_i] if slot_i < len(feature_names) else f"slot{slot_i}"
        params = (
            feature_params[slot_i]
            if slot_i < len(feature_params)
            else FEATURE_PARAMS[SPATIAL_PRIMITIVES[0]]
        )
        user_content = build_slot_hypothesis_user_content(
            task_id,
            task,
            test_index,
            feat_name,
            task_body=task_body,
            compact=ARC_FAST_INFERENCE,
        )
        prompt = _wrap_arc_chat_prompt(
            tokenizer,
            user_content,
            system_content=system_used,
            enable_thinking=hyp_thinking,
        )

        slot_temp = (
            float(ARC_GENERATION_TEMPERATURE)
            if ARC_FAST_INFERENCE
            else float(params.get("temperature", 0.7))
        )
        sp = SamplingParams(
            temperature=slot_temp,
            top_p=float(params.get("top_p", 0.92)),
            max_tokens=int(hyp_max_tokens),
            repetition_penalty=float(params.get("repetition_penalty", 1.03)),
        )
        sp = _apply_arc_guided_decoding(sp)
        setattr(sp, "stream_label", f"MBR-hypothesis/slot{slot_i}/{feat_name}")
        prompts.append(prompt)
        sp_list.append(sp)
        slot_meta.append((slot_i, feat_name, params))
        n_in = count_prompt_tokens(tokenizer, prompt)
        total_prompt_tok += n_in
        max_needed = max(max_needed, n_in + int(hyp_max_tokens))

    if max_needed > engine_ctx:
        need_ctx = int(math.ceil(max_needed / 256.0) * 256)
        raise ValueError(
            f"ARC hypothesis needs {max_needed} tokens (input+output) but vLLM "
            f"max_model_len={engine_ctx}. Restart kernel and set "
            f"VLLM_MAX_MODEL_LEN={need_ctx} at the top of the script "
            f"(auto budget is {arc_vllm_context_budget()}). "
            f"Qwen native context is 150k+; only GPU KV cache limits vLLM."
        )

    avg_prompt_tok = total_prompt_tok // max(1, k)
    slot_batch = arc_phase1_slot_batch_size(k)
    parallel_tag = (
        f"batch={slot_batch}"
        if ARC_PHASE1_PROMPT_PARALLELISM and slot_batch > 1
        else "sequential=1 slot/gen"
    )
    arc_eval_log(
        f"[ARC-PHASE-1] Hypothesis pool: {k} text proposals ({parallel_tag}; "
        f"Rendering prompts: {k}/{k} = slot count, NOT voters)"
    )
    arc_eval_log(
        f"[ARC-PHASE-1] Generate start: max_out={hyp_max_tokens}/slot | thinking={hyp_thinking} | "
        f"prompt~{avg_prompt_tok}tok/slot ({total_prompt_tok} in total) | "
        f"greedy={ARC_FAST_INFERENCE and ARC_GENERATION_TEMPERATURE == 0.0} — "
        f"first slot may take 30-120s on L4; vLLM tqdm stays 0/{k} until one slot finishes"
    )
    t_gen = time.perf_counter()
    outs = _arc_phase1_generate_slots(
        vllm_llm, prompts, sp_list, label="phase1-hyp"
    )
    arc_eval_log(
        f"[ARC-PHASE-1] Generate done: {k}/{k} slots in {time.perf_counter() - t_gen:.1f}s"
    )
    for out, (slot_i, feat_name, _params), prompt in zip(outs, slot_meta, prompts):
        gen_ids: List[int] = list(out.outputs[0].token_ids) if out.outputs else []
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        hyp_text = _extract_transformation_hypothesis(raw_text)

        rec = {
            "slot": slot_i,
            "feature": feat_name,
            "feature_index": feature_indices[slot_i] if slot_i < len(feature_indices) else slot_i,
            "text": hyp_text or raw_text.strip() or "(empty hypothesis)",
            "raw_text": raw_text,
            "num_tokens": len(gen_ids),
            "prompt": prompt,
        }
        hypotheses.append(rec)
        if verbose or (STREAM_ALL_OUTPUT and not ARC_FAST_INFERENCE):
            stream_log(
                f"  [HYPOTHESIS slot {slot_i}] {feat_name} ({len(gen_ids)} tok):\n"
                f"    {rec['text'][:500]}{'...' if len(rec['text']) > 500 else ''}"
            )

    hyp_tok = sum(int(h.get("num_tokens", 0)) for h in hypotheses)
    arc_eval_log(
        f"[ARC-PHASE-1] Hypothesis pool done: {len(hypotheses)} proposals, "
        f"{hyp_tok} output tokens (feeds Phase 2)"
    )
    return hypotheses


def pixel_wise_majority_vote_grids(
    grid_hypotheses: List[Dict[str, Any]],
    *,
    task: Optional[Dict[str, Any]] = None,
    test_index: int = 0,
) -> Tuple[Optional[List[List[int]]], Dict[str, Any]]:
    """
    Step 4 Phase 2: per-cell plurality over parsed grid hypotheses (no LLM synthesis).
    """
    parsed: List[List[List[int]]] = []
    slot_ids: List[int] = []
    for rec in grid_hypotheses:
        grid = rec.get("parsed_grid")
        if grid is not None and _is_valid_arc_grid(grid):
            parsed.append(grid)
            slot_ids.append(int(rec.get("slot", len(slot_ids))))

    if not parsed:
        return None, {
            "method": "pixel_majority",
            "n_slots": len(grid_hypotheses),
            "n_parsed": 0,
            "target_shape": None,
            "vote_counts": {},
            "tie": True,
        }

    shape_counts = Counter((len(g), len(g[0]) if g and g[0] else 0) for g in parsed)
    target_h, target_w = shape_counts.most_common(1)[0][0]
    eligible = [
        g for g in parsed if len(g) == target_h and (len(g[0]) if g else 0) == target_w
    ]
    if not eligible:
        eligible = parsed
        target_h = max(len(g) for g in eligible)
        target_w = max(len(g[0]) if g else 0 for g in eligible)

    stacked = np.asarray(eligible, dtype=np.int32)
    winners = np.zeros((target_h, target_w), dtype=np.int32)
    cell_votes: Dict[str, int] = {}
    for r in range(target_h):
        for c in range(target_w):
            col = stacked[:, r, c]
            vals, counts = np.unique(col, return_counts=True)
            winners[r, c] = int(vals[int(counts.argmax())])
            cell_votes[f"{r},{c}"] = int(col.shape[0])
    result = winners.tolist()

    return result, {
        "method": "pixel_majority",
        "n_slots": len(grid_hypotheses),
        "n_parsed": len(parsed),
        "n_eligible": len(eligible),
        "target_shape": [target_h, target_w],
        "shape_histogram": dict(shape_counts),
        "tie": False,
    }


def collect_spatial_grid_hypotheses(
    vllm_llm: Any,
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int = 0,
    *,
    k: Optional[int] = None,
    allocator: Optional[PermutationFeatureSlotAllocator] = None,
    seed: int = 42,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Step 4 Phase 1: K spatial-primitive prompts each emit one JSON grid hypothesis."""
    k = arc_hypothesis_k() if k is None else int(k)
    random.seed(seed)
    if allocator is None:
        allocator = PermutationFeatureSlotAllocator(
            internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=k
        )
    pooled = make_pooled_state([], 0, tokenizer=tokenizer)
    alloc_out = allocator(pooled, 0)
    feature_indices = alloc_out["feature_indices"]
    feature_names = alloc_out["feature_names"]
    feature_params = allocator.get_feature_params(feature_indices)

    grid_thinking = arc_resolve_enable_thinking() is True
    system_content = (
        "You solve ARC-AGI grid puzzles using spatial transformations. "
        "Output exactly one JSON 2D array of integers (0-9). "
        "No markdown, no prose, no thinking tags."
    )

    hypotheses: List[Dict[str, Any]] = []
    engine_ctx = get_inference_max_context(vllm_llm)
    slot_max_tokens = arc_spatial_slot_max_tokens(task, test_index)
    probe_primitive = feature_names[0] if feature_names else SPATIAL_PRIMITIVES[0]
    task_body = ""
    system_used = system_content
    body_fmt = "minified"

    prompts: List[str] = []
    sp_list: List[Any] = []
    slot_meta: List[Tuple[int, str, Dict[str, Any]]] = []
    max_needed = engine_ctx + 1
    task_for_prompt = task
    n_train_full = len(task.get("train") or [])
    if _arc_task_needs_compact_prompt(task, test_index):
        train_limits = [0] + [
            n for n in (1, 2, 3, n_train_full) if 0 < n <= n_train_full
        ]
    else:
        train_limits = [n_train_full] + [
            n for n in (3, 2, 1, 0) if n < n_train_full
        ]
    fitted = False
    shrink_note = ""
    arc_eval_log(f"phase1 prompt fit start ({task_id})")

    for n_train in train_limits:
        task_for_prompt = _arc_task_with_train_limit(task, n_train)
        slot_max_tokens = arc_spatial_slot_max_tokens(task_for_prompt, test_index)
        cached_body_key: Optional[Tuple[str, str]] = None
        max_prompt_in = 0
        total_prompt_tok = 0
        for _shrink in range(10):
            task_body, system_used, _probe_tok, body_fmt = (
                resolve_arc_spatial_task_body(
                    tokenizer,
                    task_id,
                    task_for_prompt,
                    test_index,
                    probe_primitive,
                    system_content=system_content,
                    engine_ctx=engine_ctx,
                    slot_max_tokens=slot_max_tokens,
                    enable_thinking=grid_thinking,
                )
            )
            body_key = (task_body, system_used)
            if body_key != cached_body_key:
                prompts = []
                sp_list = []
                slot_meta = []
                max_prompt_in = 0
                total_prompt_tok = 0
                for slot_i in range(k):
                    prim = (
                        feature_names[slot_i]
                        if slot_i < len(feature_names)
                        else f"slot{slot_i}"
                    )
                    params = (
                        feature_params[slot_i]
                        if slot_i < len(feature_params)
                        else FEATURE_PARAMS[SPATIAL_PRIMITIVES[0]]
                    )
                    user_content = build_spatial_grid_user_content(
                        task_id,
                        task_for_prompt,
                        test_index,
                        prim,
                        task_body=task_body,
                        compact=ARC_FAST_INFERENCE,
                    )
                    prompt = _wrap_arc_chat_prompt(
                        tokenizer,
                        user_content,
                        system_content=system_used,
                        enable_thinking=grid_thinking,
                        assistant_prefill=ARC_ASSISTANT_PREFILL,
                    )
                    prompts.append(prompt)
                    slot_meta.append((slot_i, prim, params))
                    n_in = count_prompt_tokens(tokenizer, prompt)
                    total_prompt_tok += n_in
                    max_prompt_in = max(max_prompt_in, n_in)
                cached_body_key = body_key
            sp_list = []
            for slot_i, prim, params in slot_meta:
                sp = SamplingParams(
                    temperature=float(ARC_GENERATION_TEMPERATURE),
                    top_p=float(params.get("top_p", 1.0)),
                    max_tokens=int(slot_max_tokens),
                    repetition_penalty=float(params.get("repetition_penalty", 1.02)),
                )
                sp = _apply_arc_guided_decoding(sp)
                setattr(sp, "stream_label", f"ARC-spatial/slot{slot_i}/{prim}")
                sp_list.append(sp)
            max_needed = max_prompt_in + int(slot_max_tokens)
            if max_needed <= engine_ctx:
                fitted = True
                if n_train < n_train_full:
                    shrink_note = f"train_pairs={n_train}/{n_train_full}"
                if _shrink > 0:
                    shrink_note = (
                        f"{shrink_note} slot_max={slot_max_tokens}".strip()
                    )
                if body_fmt == "minimal":
                    shrink_note = f"{shrink_note} body=minimal".strip()
                break
            slot_max_tokens = max(128, int(slot_max_tokens * 0.68))
        if fitted:
            break

    if not fitted:
        need_ctx = int(math.ceil(max_needed / 256.0) * 256)
        raise ValueError(
            f"ARC spatial grid needs {max_needed} tokens but vLLM "
            f"max_model_len={engine_ctx}. Set VLLM_MAX_MODEL_LEN={need_ctx} "
            f"or raise DIFFUSION_MAX_MODEL_LEN."
        )
    if shrink_note:
        arc_eval_log(
            f"[ARC-PHASE-1] Prompt budget fit: {shrink_note} "
            f"(need={max_needed}/{engine_ctx})"
        )

    slot_batch = arc_phase1_slot_batch_size(k)
    guided_on = _arc_guided_json_enabled()
    parallel_tag = (
        f"batch={slot_batch}"
        if ARC_PHASE1_PROMPT_PARALLELISM and slot_batch > 1
        else "sequential=1 slot/gen"
    )
    arc_eval_log(
        f"[ARC-PHASE-1] Spatial grid pool: {k} JSON grid hypotheses "
        f"(primitives, guided={guided_on}, thinking=False, {parallel_tag})"
    )
    arc_eval_log(
        f"[ARC-PHASE-1] Generate start: max_out={slot_max_tokens}/slot | "
        f"prompt~{total_prompt_tok // max(1, k)}tok/slot | {parallel_tag}"
    )
    arc_eval_log(
        f"phase1 generate 0/{k} slots ({parallel_tag}, ~"
        f"{total_prompt_tok // max(1, k)}tok/slot)"
    )
    t_gen = time.perf_counter()
    outs = _arc_phase1_generate_slots(
        vllm_llm, prompts, sp_list, label="phase1-spatial"
    )
    arc_eval_log(
        f"[ARC-PHASE-1] Generate done: {k}/{k} spatial slots in "
        f"{time.perf_counter() - t_gen:.1f}s"
    )

    for out, (slot_i, prim, _params), prompt in zip(outs, slot_meta, prompts):
        gen_ids: List[int] = list(out.outputs[0].token_ids) if out.outputs else []
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = parse_arc_answer_grid(raw_text)

        rec = {
            "slot": slot_i,
            "feature": prim,
            "primitive": prim,
            "feature_index": feature_indices[slot_i] if slot_i < len(feature_indices) else slot_i,
            "text": format_grid_json(parsed) if parsed else raw_text.strip(),
            "raw_text": raw_text,
            "parsed_grid": parsed,
            "num_tokens": len(gen_ids),
            "prompt": prompt,
        }
        hypotheses.append(rec)
        if verbose or (STREAM_ALL_OUTPUT and not ARC_FAST_INFERENCE):
            stream_log(
                f"  [SPATIAL slot {slot_i}] {prim} ({len(gen_ids)} tok) -> "
                f"{_grid_shape_label(parsed) if parsed else 'UNPARSED'}"
            )

    hyp_tok = sum(int(h.get("num_tokens", 0)) for h in hypotheses)
    n_parsed = sum(1 for h in hypotheses if h.get("parsed_grid") is not None)
    arc_eval_log(
        f"[ARC-PHASE-1] Spatial pool done: {n_parsed}/{len(hypotheses)} parsed grids, "
        f"{hyp_tok} output tokens"
    )
    return hypotheses


def arc_spatial_grid_ensemble_pipeline(
    vllm_llm: Any,
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int = 0,
    *,
    k: Optional[int] = None,
    allocator: Optional[PermutationFeatureSlotAllocator] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Recovery plan ARC path:
      Phase 1 — 8 spatial-primitive prompts -> 8 JSON grid hypotheses.
      Phase 2 — pixel-wise majority vote (no LLM text synthesis).
    """
    k = arc_hypothesis_k() if k is None else int(k)
    grid_hypotheses = collect_spatial_grid_hypotheses(
        vllm_llm,
        tokenizer,
        task_id,
        task,
        test_index,
        k=k,
        allocator=allocator,
        seed=seed,
        verbose=verbose,
    )
    hyp_token_total = sum(int(h.get("num_tokens", 0)) for h in grid_hypotheses)
    pooled_grid, vote_meta = pixel_wise_majority_vote_grids(
        grid_hypotheses, task=task, test_index=test_index
    )

    arc_eval_log(
        f"[ARC-PHASE-2] Pixel majority vote: {vote_meta.get('n_parsed', 0)}/"
        f"{vote_meta.get('n_slots', k)} grids -> shape {vote_meta.get('target_shape')}"
    )

    return {
        "generation_mode": "spatial_grid_ensemble+pixel_majority",
        "hypotheses": grid_hypotheses,
        "final_prompt": "",
        "final_text": format_grid_json(pooled_grid),
        "generated_text": format_grid_json(pooled_grid),
        "generated_ids": [],
        "parsed_grid": pooled_grid,
        "num_tokens": hyp_token_total,
        "hypothesis_tokens": hyp_token_total,
        "grid_tokens": 0,
        "final_output_budget": 0,
        "output_budget_cap": int(ARC_MBR_OUTPUT_TOKEN_BUDGET),
        "prompt_tokens": 0,
        "num_superblocks": 1 + k,
        "total_accepted": hyp_token_total,
        "avg_accepted_per_block": float(hyp_token_total),
        "feature_reallocations": 0,
        "acceptance_history": [1.0] if pooled_grid else [0.0],
        "feature_history": [[h.get("primitive", h.get("feature", "?")) for h in grid_hypotheses]],
        "reframe_events": 0,
        "vote_meta": vote_meta,
    }


def arc_mbr_hypothesis_pipeline(
    vllm_llm: Any,
    tokenizer: Any,
    task_id: str,
    task: Dict[str, Any],
    test_index: int = 0,
    *,
    k: Optional[int] = None,
    allocator: Optional[PermutationFeatureSlotAllocator] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Two-phase ARC MBR:
      spatial ensemble (default): K grid hypotheses + pixel majority vote.
      legacy text mode: K text hypotheses + one LLM final grid pass.
    """
    if ARC_SPATIAL_GRID_ENSEMBLE:
        return arc_spatial_grid_ensemble_pipeline(
            vllm_llm,
            tokenizer,
            task_id,
            task,
            test_index=test_index,
            k=k,
            allocator=allocator,
            seed=seed,
            verbose=verbose,
        )
    k = arc_hypothesis_k() if k is None else int(k)
    hypotheses = collect_feature_slot_hypotheses(
        vllm_llm,
        tokenizer,
        task_id,
        task,
        test_index,
        k=k,
        allocator=allocator,
        seed=seed,
        verbose=verbose,
    )
    hyp_token_total = sum(int(h.get("num_tokens", 0)) for h in hypotheses)

    engine_ctx = get_inference_max_context(vllm_llm)
    final_max_tokens = arc_mbr_final_output_budget(task, hyp_token_total)
    final_thinking = ARC_FINAL_ENABLE_THINKING and not ARC_DISABLE_THINKING
    final_prefill = "" if final_thinking else ARC_ASSISTANT_PREFILL
    if final_thinking:
        if ARC_STRUCTURED_THINKING:
            final_system = (
                "You solve ARC-AGI grid puzzles. Specialist hypotheses are provided. "
                "Inside </think> use numbered steps; after each step write HYPOTHESIS_GRID: "
                "with a JSON 2D array (partial guesses allowed). After </think> output "
                "exactly one final JSON 2D array of integers (0-9). No markdown."
            )
        else:
            final_system = (
                "You solve ARC-AGI grid puzzles. Synthesize the specialist hypotheses "
                "inside </think>, then after </think> output exactly one JSON 2D array "
                "of integers (0-9). No markdown."
            )
        if ARC_FAST_INFERENCE:
            final_system = (
                "ARC solver. Think in </think>, refine HYPOTHESIS_GRID each step, "
                "then output one JSON 2D int array (0-9) after </think>."
            )
    elif ARC_FAST_INFERENCE:
        final_system = (
            "ARC solver. Synthesize hypotheses, output one JSON 2D int array (0-9) only."
        )
    else:
        final_system = (
            "You solve ARC-AGI grid puzzles. You are given parallel textual hypotheses from "
            "specialist analysts. Synthesize the best rule, then output exactly one JSON 2D "
            "array of integers (0-9). No markdown fences, no prose after the array."
        )

    final_prompt, final_in, body_tok, body_fmt = resolve_arc_final_prompt_bundle(
        tokenizer,
        task_id,
        task,
        test_index,
        hypotheses,
        system_content=final_system,
        engine_ctx=engine_ctx,
        final_output_tokens=final_max_tokens,
        enable_thinking=final_thinking,
        assistant_prefill=final_prefill,
    )

    ctx_allows = max(64, engine_ctx - final_in - 32)
    if final_max_tokens > ctx_allows:
        stream_log(
            f"[MBR-WARN] output budget wants {final_max_tokens} tok for final grid "
            f"but engine_ctx={engine_ctx} with prompt={final_in}tok only allows "
            f"{ctx_allows}tok output. Increase VLLM_MAX_MODEL_LEN (auto={arc_vllm_context_budget()})."
        )
        final_max_tokens = ctx_allows

    arc_eval_log(
        f"[ARC-BUDGET] total_out={ARC_MBR_OUTPUT_TOKEN_BUDGET} | "
        f"phase1_hyp_used={hyp_token_total} | phase2_final_allocated={final_max_tokens} "
        f"(grid_need~{arc_final_grid_max_tokens(task)})"
    )
    arc_eval_log(
        f"[ARC-PHASE-2] Final grid: 1 synthesis prompt "
        f"(watch for 'Rendering prompts: 1/1') | prompt={final_in}tok | "
        f"max_out={final_max_tokens} | thinking={final_thinking} | prefill={final_prefill!r}"
    )
    if verbose or STREAM_ALL_OUTPUT:
        stream_log(
            f"[MBR-FINAL] detail: body={body_tok}tok({body_fmt}) engine_ctx={engine_ctx}"
        )

    grid_res = arc_direct_generate(
        vllm_llm,
        tokenizer,
        final_prompt,
        max_new_tokens=final_max_tokens,
        temperature=ARC_GENERATION_TEMPERATURE,
    )
    grid_tokens = int(grid_res.get("num_tokens", 0))
    final_prompt_tokens = int(grid_res.get("prompt_tokens", 0))

    return {
        "generation_mode": "hypothesis_slots+final_grid",
        "hypotheses": hypotheses,
        "final_prompt": final_prompt,
        "final_text": grid_res.get("final_text", final_prompt),
        "generated_text": grid_res.get("generated_text", ""),
        "generated_ids": grid_res.get("generated_ids", []),
        "parsed_grid": grid_res.get("parsed_grid"),
        "num_tokens": hyp_token_total + grid_tokens,
        "hypothesis_tokens": hyp_token_total,
        "grid_tokens": grid_tokens,
        "final_output_budget": final_max_tokens,
        "output_budget_cap": int(ARC_MBR_OUTPUT_TOKEN_BUDGET),
        "prompt_tokens": final_prompt_tokens,
        "num_superblocks": 1 + k,
        "total_accepted": grid_tokens,
        "avg_accepted_per_block": float(grid_tokens),
        "feature_reallocations": 0,
        "acceptance_history": [1.0] if grid_tokens else [0.0],
        "feature_history": [[h["feature"] for h in hypotheses]],
        "reframe_events": 0,
    }


def arc_direct_generate(
    vllm_llm: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    *,
    temperature: float = ARC_GENERATION_TEMPERATURE,
    seed: int = 42,
) -> Dict[str, Any]:
    """Single-pass generation for ARC eval final grid synthesis."""
    del seed  # temperature 0 is deterministic; seed reserved for API compatibility
    sp = SamplingParams(
        temperature=float(temperature),
        top_p=1.0,
        max_tokens=int(max_new_tokens),
    )
    sp = _apply_arc_guided_decoding(sp)
    setattr(sp, "stream_label", "MBR-FINAL-GRID")
    prompt_tokens = count_prompt_tokens(tokenizer, prompt)
    if STREAM_ALL_OUTPUT and not ARC_FAST_INFERENCE:
        stream_log(
            f"[ARC] direct generate begin (max_new_tokens={max_new_tokens}, "
            f"temp={temperature}, thinking={arc_resolve_enable_thinking()}, "
            f"guided={getattr(sp, 'arc_guided_json', False)}, "
            f"prompt_tokens={prompt_tokens})"
        )
    out = _vllm_generate_arc(vllm_llm, [prompt], [sp])[0]
    gen_ids: List[int] = []
    if out.outputs:
        gen_ids = list(out.outputs[0].token_ids)
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    result: Dict[str, Any] = {
        "final_text": prompt + generated_text,
        "generated_text": generated_text,
        "generated_ids": gen_ids,
        "num_tokens": len(gen_ids),
        "prompt_tokens": prompt_tokens,
        "num_superblocks": 1,
        "total_accepted": len(gen_ids),
        "avg_accepted_per_block": float(len(gen_ids)),
        "feature_reallocations": 0,
        "acceptance_history": [1.0] if gen_ids else [0.0],
        "feature_history": [["DirectGenerate"]],
        "reframe_events": 0,
        "generation_mode": "direct",
    }
    gen_suffix = extract_arc_generated_suffix(result, prompt)
    parsed_grid = parse_arc_answer_grid(gen_suffix)
    if parsed_grid is not None:
        result["parsed_grid"] = parsed_grid
        result["generated_text"] = json.dumps(parsed_grid, separators=(",", ":"))
    return result


def parse_grid_from_text(
    text: str, *, only_after: Optional[str] = "Test output"
) -> Optional[List[List[int]]]:
    """Best-effort extraction of a 2D integer grid from model text."""
    search_text = _strip_model_artifacts(text or "")
    if only_after and only_after in search_text:
        search_text = search_text.split(only_after, 1)[-1]

    seen_raw: set = set()
    candidates: List[Tuple[int, List[List[int]]]] = []
    for variant in _grid_attempt_text_variants(search_text):
        # Explicit [[ anchors — do not stop at the first broken attempt.
        for anchor in re.finditer(r"\[\[", variant):
            raw = _balanced_bracket_span_at(variant, anchor.start())
            if not raw or raw in seen_raw:
                continue
            seen_raw.add(raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if _is_valid_arc_grid(parsed):
                candidates.append((anchor.start(), parsed))

        for raw in _balanced_bracket_spans(variant):
            if raw in seen_raw:
                continue
            seen_raw.add(raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if _is_valid_arc_grid(parsed):
                candidates.append((variant.rfind(raw), parsed))

    if not candidates:
        for variant in _grid_attempt_text_variants(search_text):
            row_grid = _parse_grid_from_row_pattern(variant)
            if row_grid is not None:
                return row_grid
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def grids_equal(pred: List[List[int]], gold: List[List[int]]) -> bool:
    return pred == gold


# Standard ARC-AGI palette (colors 0–9)
ARC_PALETTE_RGB: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (240, 18, 190),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
]


def _arc_grade_output_dir() -> str:
    out = "/kaggle/working/arc_grades" if _on_kaggle() else "arc_grades"
    os.makedirs(out, exist_ok=True)
    return out


def grid_cell_stats(
    pred: Optional[List[List[int]]], gold: List[List[int]]
) -> Dict[str, Any]:
    """Compare pred vs gold; report match rate and shape mismatch."""
    gh, gw = len(gold), len(gold[0]) if gold else 0
    if pred is not None and not _is_valid_arc_grid(pred):
        pred = None
    if pred is None:
        return {
            "parsed": False,
            "correct": False,
            "gold_shape": (gh, gw),
            "pred_shape": None,
            "matching_cells": 0,
            "gold_cells": gh * gw,
            "match_rate": 0.0,
            "shape_match": False,
        }
    ph, pw = len(pred), len(pred[0]) if pred else 0
    shape_match = (ph, pw) == (gh, gw)
    matching = 0
    if shape_match:
        for r in range(ph):
            for c in range(pw):
                if pred[r][c] == gold[r][c]:
                    matching += 1
    total = gh * gw
    return {
        "parsed": True,
        "correct": shape_match and matching == total,
        "gold_shape": (gh, gw),
        "pred_shape": (ph, pw),
        "matching_cells": matching,
        "gold_cells": total,
        "match_rate": matching / max(1, total),
        "shape_match": shape_match,
    }


def _grade_verdict_label(stats: Dict[str, Any]) -> str:
    if not stats["parsed"]:
        return "UNPARSED"
    if stats["correct"]:
        return "PASS"
    if not stats["shape_match"]:
        return "FAIL (shape)"
    return "FAIL"


def _ansi_cell(value: int, glyph: str = "  ") -> str:
    value = max(0, min(9, int(value)))
    r, g, b = ARC_PALETTE_RGB[value]
    return f"\033[48;2;{r};{g};{b}m{glyph}\033[0m"


def _ansi_diff_cell(pred_val: Optional[int], gold_val: int) -> str:
    if pred_val is None:
        return "\033[48;2;64;64;64m??\033[0m"
    if pred_val == gold_val:
        return _ansi_cell(gold_val, "OK")
    return "\033[48;2;180;0;0mXX\033[0m"


def _render_grid_ascii(
    grid: List[List[int]],
    *,
    title: str,
    diff_against: Optional[List[List[int]]] = None,
    pred_for_diff: Optional[List[List[int]]] = None,
) -> List[str]:
    lines = [title]
    if not grid:
        lines.append("  (empty)")
        return lines
    for r, row in enumerate(grid):
        cells = []
        for c, val in enumerate(row):
            if diff_against is not None:
                if (
                    pred_for_diff is not None
                    and r < len(pred_for_diff)
                    and c < len(pred_for_diff[r])
                ):
                    pv = pred_for_diff[r][c]
                else:
                    pv = None
                gv = diff_against[r][c] if r < len(diff_against) and c < len(diff_against[r]) else 0
                cells.append(_ansi_diff_cell(pv, gv))
            else:
                cells.append(_ansi_cell(val))
        lines.append("  " + "".join(cells))
    return lines


def _align_grid_columns(blocks: List[List[str]]) -> List[str]:
    """Place multiple ASCII grid blocks side-by-side."""
    if not blocks:
        return []
    heights = [len(b) for b in blocks]
    max_h = max(heights)
    padded = [b + [""] * (max_h - len(b)) for b in blocks]
    merged: List[str] = []
    for row_idx in range(max_h):
        merged.append("    ".join(p[row_idx] for p in padded))
    return merged


def format_grid_json(grid: Optional[List[List[int]]]) -> str:
    """Compact JSON for terminal answer comparison."""
    if grid is None:
        return "(unparsed — model did not return a valid 2D integer grid)"
    return json.dumps(grid, separators=(",", ":"))


def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    """Token length of a prompt (for prefill-aware throughput reporting)."""
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(prompt, add_special_tokens=True))
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    return int(encoded["input_ids"].shape[1])


def generation_timing_stats(
    elapsed_s: float,
    num_tokens: int,
    *,
    prompt_tokens: int = 0,
) -> Dict[str, Any]:
    """Elapsed wall time + output/effective tokens/sec for one generation call."""
    elapsed = max(0.0, float(elapsed_s))
    tokens = max(0, int(num_tokens or 0))
    prompt_tok = max(0, int(prompt_tokens or 0))
    total_tok = prompt_tok + tokens
    decode_tps = tokens / max(elapsed, 1e-6)
    effective_tps = total_tok / max(elapsed, 1e-6)
    return {
        "elapsed_s": elapsed,
        "num_tokens": tokens,
        "prompt_tokens": prompt_tok,
        "total_tokens": total_tok,
        "tps": decode_tps,
        "decode_tps": decode_tps,
        "effective_tps": effective_tps,
    }


def format_timing_line(timing: Dict[str, Any]) -> str:
    return (
        f"{timing['num_tokens']} out tok in {timing['elapsed_s']:.2f}s "
        f"(decode {timing.get('decode_tps', timing['tps']):.1f} tok/s)"
    )


def format_timing_detail_line(timing: Dict[str, Any]) -> str:
    """Full timing: prompt + output tokens and effective throughput."""
    prompt_tok = timing.get("prompt_tokens", 0)
    return (
        f"{timing['num_tokens']} out + {prompt_tok} prompt tok in "
        f"{timing['elapsed_s']:.2f}s | decode "
        f"{timing.get('decode_tps', timing['tps']):.1f} tok/s | effective "
        f"{timing.get('effective_tps', timing['tps']):.1f} tok/s"
    )


def arc_eval_log(message: str) -> None:
    """Print + flush ARC eval progress lines."""
    print(message, flush=True)


def print_arc_answer_comparison(
    *,
    task_id: str,
    task_idx: int,
    num_tasks: int,
    test_index: int,
    gold: List[List[int]],
    mbr_pred: Optional[List[List[int]]],
    mbr_stats: Dict[str, Any],
    mbr_raw_text: Optional[str] = None,
    mbr_timing: Optional[Dict[str, Any]] = None,
    split: str = ARC_DATA_SPLIT,
) -> None:
    """Print ground truth and MBR answer side-by-side (JSON grids)."""
    bar = "=" * 80
    print(f"\n{bar}")
    print(
        f"ARC ANSWER vs GROUND TRUTH  |  task {task_idx + 1}/{num_tasks}  "
        f"|  id={task_id}  |  split={split}  |  test #{test_index}"
    )
    print(bar)

    print("\nGROUND TRUTH (gold labels):")
    print(format_grid_json(gold))

    m_tag = _grade_verdict_label(mbr_stats)
    m_time = (
        f"  |  {format_timing_line(mbr_timing)}"
        if mbr_timing
        else ""
    )
    print(f"\nMILLION-BRAINS PREDICTION [{m_tag}] "
          f"({mbr_stats['matching_cells']}/{mbr_stats['gold_cells']} cells, "
          f"{mbr_stats['match_rate'] * 100:.1f}% match){m_time}:")
    print(format_grid_json(mbr_pred))
    if mbr_pred is None and mbr_raw_text:
        tail = mbr_raw_text[-800:] if len(mbr_raw_text) > 800 else mbr_raw_text
        print("MBR raw model output (tail):")
        print(tail)

    if mbr_pred and gold:
        print("\nCELL-BY-CELL (MBR vs gold) — XX=mismatch, ok=match:")
        for r in range(len(gold)):
            row_markers = []
            for c in range(len(gold[r])):
                if r >= len(mbr_pred) or c >= len(mbr_pred[r]):
                    row_markers.append("??")
                elif mbr_pred[r][c] == gold[r][c]:
                    row_markers.append("ok")
                else:
                    row_markers.append("XX")
            print(f"  row {r}: " + " ".join(row_markers))

    print(bar, flush=True)


def print_arc_final_result(
    *,
    task_id: str,
    task_idx: int,
    num_tasks: int,
    test_index: int,
    test_input: List[List[int]],
    gold: List[List[int]],
    mbr_pred: Optional[List[List[int]]],
    mbr_stats: Dict[str, Any],
    mbr_timing: Optional[Dict[str, Any]] = None,
    split: str = ARC_DATA_SPLIT,
) -> None:
    """Single end-of-test summary: verdict, timing, and final gold vs MBR prediction."""
    m_verdict = _grade_verdict_label(mbr_stats)
    m_icon = "PASS" if mbr_stats["correct"] else "FAIL"
    m_time = (
        format_timing_detail_line(mbr_timing) if mbr_timing else "n/a"
    )
    in_shape = _grid_shape_label(test_input)

    bar = "═" * 78
    print(f"\n{bar}")
    print(
        f"ARC FINAL  task {task_idx + 1}/{num_tasks}  id={task_id}  "
        f"split={split}  test=#{test_index}  input={in_shape}"
    )
    print(
        f"  MBR [{m_icon}] {m_verdict:<12}  "
        f"{mbr_stats['matching_cells']}/{mbr_stats['gold_cells']} cells  |  {m_time}"
    )
    print(bar)

    blocks = [
        _render_grid_ascii(gold, title=f"GOLD ({_grid_shape_label(gold)})"),
    ]
    if mbr_pred:
        blocks.append(
            _render_grid_ascii(
                mbr_pred,
                title=f"MBR [{m_verdict}]",
                diff_against=gold,
                pred_for_diff=mbr_pred,
            )
        )
    else:
        blocks.append(["MBR [UNPARSED]", "  (no valid grid)"])
    for line in _align_grid_columns(blocks):
        print("  " + line)
    print(bar, flush=True)


def print_arc_full_answer_report(
    comparisons: List[Dict[str, Any]], split: str
) -> None:
    """Final digest: every task's MBR predictions vs ground truth."""
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"ARC FULL ANSWER REPORT  ({split} split — all tasks vs ground truth)")
    print(bar)
    print(
        f"{'Task':<14} {'Test':>4}  {'MBR':<10}  {'MBR TPS':>12}  Gold"
    )
    print("-" * 80)
    for rec in comparisons:
        gold_s = rec.get("gold_json", "[]")
        if len(gold_s) > 36:
            gold_s = gold_s[:33] + "..."
        m_timing = rec.get("mbr_timing") or {}
        m_tps = f"{m_timing.get('tps', 0.0):.2f} tok/s"
        m_elapsed = f"{m_timing.get('elapsed_s', 0.0):.1f}s"
        print(
            f"{rec['task_id']:<14} {rec['test_index']:>4}  "
            f"{rec['mbr_verdict']:<10}  "
            f"{m_tps:>6} ({m_elapsed:>5})  {gold_s}"
        )
        mbr_s = rec.get("mbr_json", "(unparsed)")
        if len(mbr_s) > 76:
            mbr_s = mbr_s[:73] + "..."
        print(f"    gold: {rec.get('gold_json', '[]')}")
        print(
            f"    mbr : {mbr_s}  "
            f"[{format_timing_line(m_timing) if m_timing else 'n/a'}]"
        )
        print("-" * 80)
    print(bar)


def print_arc_grade_card(
    *,
    task_id: str,
    task_idx: int,
    num_tasks: int,
    test_index: int,
    test_input: List[List[int]],
    gold: List[List[int]],
    mbr_pred: Optional[List[List[int]]],
    mbr_stats: Dict[str, Any],
    split: str = ARC_DATA_SPLIT,
) -> None:
    """Terminal grade card: test input vs gold answer key vs MBR prediction."""
    m_verdict = _grade_verdict_label(mbr_stats)
    m_icon = "✓" if mbr_stats["correct"] else "✗"

    bar = "═" * 78
    print(f"\n╔{bar}╗")
    print(
        f"║  ARC GRADE  task {task_idx + 1}/{num_tasks}  id={task_id}  "
        f"split={split}  test=#{test_index}"
        f"{' ' * max(0, 18 - len(task_id))}║"
    )
    print(
        f"║  MILLION-BRAINS {m_icon} {m_verdict:<14}"
        f"{' ' * 40}║"
    )
    print(f"╠{bar}╣")

    blocks = [
        _render_grid_ascii(test_input, title="TEST INPUT (challenge)"),
        _render_grid_ascii(gold, title="GOLD SOLUTION (test set labels)"),
        _render_grid_ascii(
            mbr_pred if mbr_pred else [[-1]],
            title=f"MBR PRED [{m_verdict}]",
        )
        if mbr_pred
        else ["MBR PRED [UNPARSED]", "  (model output not a valid grid)"],
    ]
    for line in _align_grid_columns(blocks):
        print(f"║ {line:<76} ║")

    print(f"╠{bar}╣")
    print(
        f"║  MBR match: {mbr_stats['matching_cells']:>3}/{mbr_stats['gold_cells']:<3} "
        f"({mbr_stats['match_rate'] * 100:5.1f}%)  "
        f"shape {mbr_stats['pred_shape'] or '—'} vs gold {mbr_stats['gold_shape']}"
        f"{' ' * 12}║"
    )
    print(f"╚{bar}╝")

    if mbr_pred and mbr_stats["shape_match"]:
        diff_lines = _align_grid_columns(
            [
                _render_grid_ascii(
                    gold,
                    title="DIFF vs GOLD (mbr)",
                    diff_against=gold,
                    pred_for_diff=mbr_pred,
                ),
            ]
        )
        print("  Diff overlay (OK=match, XX=mismatch, ??=missing):")
        for line in diff_lines:
            print(f"  {line}")


def _draw_grid_matplotlib(ax, grid: List[List[int]], title: str) -> None:
    import numpy as np

    if not _is_valid_arc_grid(grid):
        ax.text(
            0.5,
            0.5,
            "INVALID\nGRID",
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        return

    arr = np.array(grid, dtype=np.int32)
    cmap_colors = [
        [r / 255, g / 255, b / 255, 1.0] for r, g, b in ARC_PALETTE_RGB
    ]
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(cmap_colors)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def _draw_diff_matplotlib(
    ax, pred: Optional[List[List[int]]], gold: List[List[int]], title: str
) -> None:
    import numpy as np

    gh, gw = len(gold), len(gold[0])
    diff = np.zeros((gh, gw, 3), dtype=float)
    for r in range(gh):
        for c in range(gw):
            gv = gold[r][c]
            if pred is None or r >= len(pred) or c >= len(pred[0]):
                diff[r, c] = (0.35, 0.35, 0.35)  # unparsed / out of bounds
            elif pred[r][c] == gv:
                pr, pg, pb = ARC_PALETTE_RGB[gv]
                diff[r, c] = (pr / 255 * 0.55, pg / 255 * 0.55, pb / 255 * 0.55)
            else:
                diff[r, c] = (0.9, 0.1, 0.1)  # wrong cell
    ax.imshow(diff, interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def save_arc_grade_image(
    *,
    task_id: str,
    test_index: int,
    test_input: List[List[int]],
    gold: List[List[int]],
    mbr_pred: Optional[List[List[int]]],
    mbr_stats: Dict[str, Any],
    train_pairs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Save a PNG grade card showing MBR prediction against the test-set gold grid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    m_verdict = _grade_verdict_label(mbr_stats)
    n_train = min(len(train_pairs or []), 3) if ARC_SHOW_TRAIN_EXAMPLES else 0
    ncols = 4 + n_train
    fig_w = max(10, 2.2 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 3.2))
    if ncols == 1:
        axes = [axes]

    col = 0
    if n_train:
        for i in range(n_train):
            pair = train_pairs[i]
            _draw_grid_matplotlib(
                axes[col],
                pair["input"],
                f"Train {i + 1} in",
            )
            col += 1

    _draw_grid_matplotlib(axes[col], test_input, "TEST INPUT")
    col += 1
    _draw_grid_matplotlib(axes[col], gold, "GOLD (test labels)")
    col += 1
    if mbr_pred and _is_valid_arc_grid(mbr_pred):
        _draw_grid_matplotlib(axes[col], mbr_pred, f"MBR [{m_verdict}]")
    elif mbr_pred:
        axes[col].text(
            0.5, 0.5, "MALFORMED\nGRID", ha="center", va="center", fontsize=12
        )
        axes[col].set_title(f"MBR [{m_verdict}]")
        axes[col].axis("off")
    else:
        axes[col].text(0.5, 0.5, "UNPARSED", ha="center", va="center", fontsize=12)
        axes[col].set_title(f"MBR [{m_verdict}]")
        axes[col].axis("off")
    col += 1
    _draw_diff_matplotlib(
        axes[col],
        mbr_pred,
        gold,
        f"MBR diff ({mbr_stats['match_rate'] * 100:.0f}%)",
    )

    m_color = "#2ecc40" if mbr_stats["correct"] else "#ff4136"
    fig.suptitle(
        f"ARC {task_id} test #{test_index}  |  MBR: {m_verdict}",
        fontsize=11,
        fontweight="bold",
        color="#222222",
    )
    fig.text(
        0.5,
        0.02,
        f"MBR {mbr_stats['matching_cells']}/{mbr_stats['gold_cells']} cells",
        ha="center",
        fontsize=9,
    )
    mbr_ax_idx = n_train + 2
    for spine in axes[mbr_ax_idx].spines.values():
        spine.set_edgecolor(m_color)
        spine.set_linewidth(3)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    out_path = os.path.join(
        _arc_grade_output_dir(), f"{task_id}_test{test_index}_grade.png"
    )
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path


def display_arc_grade_image(image_path: str) -> None:
    """Show grade PNG inline when running in a Jupyter/Kaggle notebook."""
    if not image_path or not os.path.isfile(image_path):
        return
    try:
        from IPython.display import Image, display

        display(Image(filename=image_path))
    except Exception:
        print(f"[ARC GRADE] saved image: {image_path}")


def grade_arc_test_case(
    *,
    task_id: str,
    task_idx: int,
    num_tasks: int,
    test_index: int,
    test_input: List[List[int]],
    gold: List[List[int]],
    mbr_pred: Optional[List[List[int]]],
    train_pairs: Optional[List[Dict[str, Any]]] = None,
    split: str = ARC_DATA_SPLIT,
) -> Dict[str, Any]:
    """Full visual grading for one test case: stats + terminal card + optional PNG."""
    mbr_stats = grid_cell_stats(mbr_pred, gold)

    if ARC_VISUAL_GRADING and not ARC_PRINT_FINAL_MATRICES:
        print_arc_grade_card(
            task_id=task_id,
            task_idx=task_idx,
            num_tasks=num_tasks,
            test_index=test_index,
            test_input=test_input,
            gold=gold,
            mbr_pred=mbr_pred,
            mbr_stats=mbr_stats,
            split=split,
        )

    image_path = None
    if ARC_VISUAL_GRADING and ARC_SAVE_GRADE_IMAGES:
        image_path = save_arc_grade_image(
            task_id=task_id,
            test_index=test_index,
            test_input=test_input,
            gold=gold,
            mbr_pred=mbr_pred,
            mbr_stats=mbr_stats,
            train_pairs=train_pairs,
        )
        if image_path:
            display_arc_grade_image(image_path)

    return {
        "mbr_stats": mbr_stats,
        "image_path": image_path,
    }


def print_arc_gradeboard(per_task: List[Dict[str, Any]], split: str) -> None:
    """Final at-a-glance scoreboard for every challenged task."""
    print("\n" + "━" * 78)
    print(f"ARC GRADEBOARD  ({split} split — MBR vs test-set gold labels)")
    print("━" * 78)
    print(f"{'Task ID':<12} {'Tests':>5}  {'Million-Brains':>16}")
    print("-" * 78)
    for rec in per_task:
        tid = rec["task_id"]
        tests = len(rec.get("mbr", []))
        m_pass = sum(1 for r in rec["mbr"] if r.get("correct"))
        m_bar = "█" * m_pass + "░" * max(0, tests - m_pass)
        print(f"{tid:<12} {tests:>5}  {m_pass}/{tests} {m_bar}")
    print("━" * 78)









































































def print_post_load_arc_config() -> None:
    """Short cheat-sheet printed right after model load (before ARC eval banner)."""
    hyp_n = arc_hypothesis_k()
    print("\n" + "-" * 72)
    print("RUNTIME ARC CONFIG — QUICK REFERENCE")
    print("-" * 72)
    print("  Engine       : single DiffusionGemma vLLM")
    print(f"  Per test     : Phase1:{hyp_n} props -> Phase2:1 grid (pixel vote if spatial ensemble)")
    print(f"  BENCHMARK_K  : {K} (demo benchmark only — NOT used in ARC Phase 1/2)")
    print(
        f"  vLLM hints   : Phase1 shows 'Rendering prompts: {hyp_n}/{hyp_n}' | "
        f"Phase2 shows 'Rendering prompts: 1/1'"
    )
    print("-" * 72 + "\n")

def print_arc_pipeline_architecture(
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
    print("\n" + "=" * 80)
    print("ARC PIPELINE — HOW TO READ THE LOGS")
    print("=" * 80)
    if ARC_SPATIAL_GRID_ENSEMBLE:
        phase1_line = (
            f"    [ARC-PHASE-1]  Spatial pool     -> {hyp_n} JSON grid hypotheses "
            f"(guided={ARC_GUIDED_JSON_DECODING}, thinking=False, {phase1_mode})\n"
        )
        phase2_line = (
            "    [ARC-PHASE-2]  Pixel majority   -> per-cell vote across parsed grids\n"
        )
    else:
        phase1_line = (
            f"    [ARC-PHASE-1]  Hypothesis pool  -> {hyp_n} parallel TEXT proposals\n"
        )
        phase2_line = (
            "    [ARC-PHASE-2]  Final grid       -> 1 JSON grid synthesis\n"
            "                   vLLM shows: Rendering prompts: 1/1\n"
        )
    print(
        "  Single engine runs TWO phases per test case:\n"
        f"{phase1_line}"
        f"                   vLLM shows: Rendering prompts: {hyp_n}/{hyp_n}  (= hypothesis slot count)\n"
        f"{phase2_line}"
        "  Note: Phase-1 can look idle at 'Processed 0/N' until the first slot fully completes."
    )
    print("")
    print(
        f"  BENCHMARK_K={K} is for the demo/benchmark path only — "
        f"ARC hypothesis count is ARC_HYPOTHESIS_SLOTS={ARC_HYPOTHESIS_SLOTS}"
    )
    print("=" * 80 + "\n")

def evaluate_arc_dataset(
    vllm_llm: Optional[Any],
    tokenizer: Any,
    challenges_path: str,
    solutions_path: str,
    *,
    max_tasks: Optional[int] = EVAL_MAX_TASKS,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    k: int = K,
    block_size: int = BLOCK_SIZE,
    enable_reallocation: bool = ENABLE_FEATURE_REALLOCATION,
    seed: int = SEED,
    split: str = ARC_DATA_SPLIT,
    visual_grading: bool = ARC_VISUAL_GRADING,
) -> Dict[str, Any]:
    """
    Run Million-Brains DiffusionGemma ARC-AGI evaluation on a split.
    Requires explicit challenges + solutions paths (typically under data/, gitignored).
    """
    dataset = load_arc_dataset(challenges_path, solutions_path)
    challenges = dataset["challenges"]
    solutions = dataset["solutions"]

    task_ids = sorted(challenges.keys())
    if EVAL_SMOKE_TASK_ID:
        if EVAL_SMOKE_TASK_ID in challenges:
            task_ids = [EVAL_SMOKE_TASK_ID]
        else:
            print(
                f"[ARC] EVAL_SMOKE_TASK_ID={EVAL_SMOKE_TASK_ID!r} not in challenges; "
                "running zero tasks."
            )
            task_ids = []
    elif max_tasks is not None:
        task_ids = task_ids[: max(0, int(max_tasks))]

    print("\n" + "=" * 80)
    print("ARC-AGI EVALUATION")
    print("=" * 80)
    print(f"Challenges: {os.path.abspath(challenges_path)}")
    print(f"Solutions:  {os.path.abspath(solutions_path)}")
    print(f"Tasks:      {len(task_ids)}")
    print(f"Split:      {split}")
    print(f"Visual grading: {visual_grading} (save images: {ARC_SAVE_GRADE_IMAGES})")
    print(f"Print all answers vs gold: {ARC_PRINT_ALL_ANSWERS}")
    print(f"ARC eval verbose (MBR internals): {ARC_EVAL_VERBOSE}")
    print(f"Stream generation: {STREAM_GENERATION} | stream all: {STREAM_ALL_OUTPUT}")
    print(f"Stream thinking: {STREAM_PRINT_THINKING} | disable_thinking={ARC_DISABLE_THINKING}")
    print(
        f"Structured thinking: {ARC_STRUCTURED_THINKING} | "
        f"step matrices: {ARC_PRINT_STEP_MATRICES} | final matrices: {ARC_PRINT_FINAL_MATRICES}"
    )
    print(
        f"Fast inference: {ARC_FAST_INFERENCE} | try vLLM: {ARC_TRY_VLLM} | "
        f"max_prompt_tok: {ARC_MAX_PROMPT_TOKENS} | "
        f"vllm_ctx_budget: {arc_vllm_context_budget()} | "
        f"engine pref: {'vLLM' if ARC_TRY_VLLM and not PREFER_HF_INFERENCE else 'HF'}"
    )
    if EVAL_SMOKE_TASK_ID:
        print(f"Smoke task only: {EVAL_SMOKE_TASK_ID}")
    print(
        f"ARC generation: chat_template={ARC_USE_CHAT_TEMPLATE}, "
        f"temp={ARC_GENERATION_TEMPERATURE}, "
        f"output_budget={ARC_MBR_OUTPUT_TOKEN_BUDGET} tok/test, "
        f"hyp_thinking={ARC_HYPOTHESIS_ENABLE_THINKING}, "
        f"final_thinking={ARC_FINAL_ENABLE_THINKING}"
    )
    print_arc_pipeline_architecture(vllm_llm=vllm_llm)
    print(f"[CONFIG] Single engine | Phase1={arc_hypothesis_k()} props/test | Phase2=1 grid/test")
    if visual_grading and ARC_SAVE_GRADE_IMAGES:
        print(f"Grade images: {_arc_grade_output_dir()}/")
    print("-" * 80)

    allocator = PermutationFeatureSlotAllocator(
        internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=k
    )

    summary = {
        "challenges_path": os.path.abspath(challenges_path),
        "solutions_path": os.path.abspath(solutions_path),
        "num_tasks": len(task_ids),
        "mbr": {"correct": 0, "parsed": 0, "tests": 0, "time": 0.0, "tokens": 0},
        "per_task": [],
        "answer_comparisons": [],
    }

    for task_idx, task_id in enumerate(task_ids):
        if task_id not in solutions:
            print(f"[ARC] Skipping {task_id}: no entry in solutions file")
            continue

        task = challenges[task_id]
        gold_tests = solutions[task_id]
        num_tests = min(len(task.get("test", [])), len(gold_tests))

        task_record = {
            "task_id": task_id,
            "mbr": [],
        }

        for test_index in range(num_tests):
            prompt = build_arc_inference_prompt(
                tokenizer, task_id, task, test_index=test_index
            )
            gold = gold_tests[test_index]
            test_input = task["test"][test_index]["input"]

            hyp_n = arc_hypothesis_k()
            phase2 = (
                "pixel majority vote"
                if ARC_SPATIAL_GRID_ENSEMBLE
                else "Phase2:1 grid"
            )
            arc_eval_log(
                f"\n[ARC] >>> task {task_idx + 1}/{len(task_ids)} {task_id} "
                f"test#{test_index} — Phase1:{hyp_n} "
                f"{'spatial grids' if ARC_SPATIAL_GRID_ENSEMBLE else 'props'} -> {phase2}"
            )
            t0 = time.perf_counter()
            mbr_res: Dict[str, Any] = {}
            if ARC_SLOT_HYPOTHESIS_MODE:
                mbr_res = arc_mbr_hypothesis_pipeline(
                    vllm_llm,
                    tokenizer,
                    task_id,
                    task,
                    test_index=test_index,
                    k=arc_hypothesis_k(),
                    allocator=allocator,
                    seed=seed + test_index,
                    verbose=(ARC_EVAL_VERBOSE or STREAM_ALL_OUTPUT)
                    and not ARC_FAST_INFERENCE,
                )
                mbr_prompt_for_extract = mbr_res.get("final_prompt", prompt)
                mbr_elapsed = time.perf_counter() - t0
                mbr_timing = generation_timing_stats(
                    mbr_elapsed,
                    mbr_res["num_tokens"],
                    prompt_tokens=mbr_res.get("prompt_tokens", 0),
                )
                mbr_answer_text = extract_arc_generated_suffix(
                    mbr_res, mbr_prompt_for_extract
                )
                mbr_pred = parse_arc_answer_grid(mbr_answer_text)
            else:
                mbr_res = million_brains_dflash_generate(
                    vllm_llm,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    k=k,
                    block_size=block_size,
                    allocator=allocator,
                    enable_reallocation=enable_reallocation,
                    seed=seed + test_index,
                    verbose=ARC_EVAL_VERBOSE or STREAM_ALL_OUTPUT,
                    arc_context={
                        "task_id": task_id,
                        "test_input": test_input,
                        "gold": gold,
                    },
                )
                mbr_prompt_for_extract = prompt
                mbr_elapsed = time.perf_counter() - t0
                mbr_timing = generation_timing_stats(
                    mbr_elapsed,
                    mbr_res["num_tokens"],
                    prompt_tokens=mbr_res.get("prompt_tokens", 0),
                )
                mbr_answer_text = extract_arc_generated_suffix(
                    mbr_res, mbr_prompt_for_extract
                )
                mbr_pred = parse_arc_answer_grid(mbr_answer_text)

            arc_eval_log(
                f"[ARC] <<< task {task_idx + 1}/{len(task_ids)} {task_id} "
                f"test#{test_index} — done | {format_timing_line(mbr_timing)} | "
                f"phase1_hyp_tok={mbr_res.get('hypothesis_tokens', 0)} "
                f"phase2_grid_tok={mbr_res.get('grid_tokens', 0)} "
                f"budget={mbr_res.get('output_budget_cap', ARC_MBR_OUTPUT_TOKEN_BUDGET)}"
            )
            summary["mbr"]["time"] += mbr_elapsed
            summary["mbr"]["tokens"] += int(mbr_res.get("num_tokens", 0))
            summary["mbr"]["tests"] += 1

            grade_info = grade_arc_test_case(
                task_id=task_id,
                task_idx=task_idx,
                num_tasks=len(task_ids),
                test_index=test_index,
                test_input=test_input,
                gold=gold,
                mbr_pred=mbr_pred,
                train_pairs=task.get("train") if ARC_SHOW_TRAIN_EXAMPLES else None,
                split=split,
            ) if visual_grading else {
                "mbr_stats": grid_cell_stats(mbr_pred, gold),
                "image_path": None,
            }

            mbr_stats = grade_info["mbr_stats"]

            if ARC_PRINT_FINAL_MATRICES:
                print_arc_final_result(
                    task_id=task_id,
                    task_idx=task_idx,
                    num_tasks=len(task_ids),
                    test_index=test_index,
                    test_input=test_input,
                    gold=gold,
                    mbr_pred=mbr_pred,
                    mbr_stats=mbr_stats,
                    mbr_timing=mbr_timing,
                    split=split,
                )

            comparison_rec = {
                "task_id": task_id,
                "task_idx": task_idx,
                "test_index": test_index,
                "split": split,
                "gold": gold,
                "gold_json": format_grid_json(gold),
                "mbr_pred": mbr_pred,
                "mbr_json": format_grid_json(mbr_pred),
                "mbr_verdict": _grade_verdict_label(mbr_stats),
                "mbr_correct": mbr_stats["correct"],
                "mbr_match_rate": mbr_stats["match_rate"],
                "mbr_timing": mbr_timing,
                "mbr_mode": mbr_res.get("generation_mode"),
            }
            summary["answer_comparisons"].append(comparison_rec)

            if ARC_PRINT_ALL_ANSWERS and not ARC_PRINT_FINAL_MATRICES:
                print_arc_answer_comparison(
                    task_id=task_id,
                    task_idx=task_idx,
                    num_tasks=len(task_ids),
                    test_index=test_index,
                    gold=gold,
                    mbr_pred=mbr_pred,
                    mbr_stats=mbr_stats,
                    mbr_raw_text=mbr_answer_text,
                    mbr_timing=mbr_timing,
                    split=split,
                )

            if mbr_stats["parsed"]:
                summary["mbr"]["parsed"] += 1
            if mbr_stats["correct"]:
                summary["mbr"]["correct"] += 1

            task_record["mbr"].append(
                {
                    "test_index": test_index,
                    "parsed": mbr_stats["parsed"],
                    "correct": mbr_stats["correct"],
                    "match_rate": mbr_stats["match_rate"],
                    "matching_cells": mbr_stats["matching_cells"],
                    "gold_cells": mbr_stats["gold_cells"],
                    "prediction": mbr_pred,
                    "gold": gold,
                    "elapsed_s": mbr_timing["elapsed_s"],
                    "num_tokens": mbr_timing["num_tokens"],
                    "tps": mbr_timing["tps"],
                }
            )
            if grade_info.get("image_path"):
                task_record.setdefault("grade_images", []).append(
                    grade_info["image_path"]
                )

        summary["per_task"].append(task_record)
        m_ok = sum(1 for r in task_record["mbr"] if r["correct"])
        m_task_tps = (
            sum(r.get("num_tokens", 0) for r in task_record["mbr"])
            / max(1e-6, sum(r.get("elapsed_s", 0.0) for r in task_record["mbr"]))
        )
        print(
            f"[ARC] {task_idx + 1:4d}/{len(task_ids)} {task_id} "
            f"mbr {m_ok}/{num_tests} ({m_task_tps:.2f} tok/s)"
        )

    def _rates(bucket: Dict[str, Any]) -> Dict[str, float]:
        tests = max(1, bucket["tests"])
        return {
            "accuracy": bucket["correct"] / tests,
            "parse_rate": bucket["parsed"] / tests,
            "tps": bucket["tokens"] / max(1e-6, bucket["time"]),
        }

    summary["mbr"].update(_rates(summary["mbr"]))
    summary["split"] = split

    if visual_grading:
        print_arc_gradeboard(summary["per_task"], split)

    if ARC_PRINT_ALL_ANSWERS and summary["answer_comparisons"]:
        print_arc_full_answer_report(summary["answer_comparisons"], split)
        report_path = ARC_ANSWER_REPORT_PATH
        if report_path is None:
            report_path = (
                "/kaggle/working/arc_answer_report.json"
                if _on_kaggle()
                else "arc_answer_report.json"
            )
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "split": split,
                        "num_comparisons": len(summary["answer_comparisons"]),
                        "comparisons": summary["answer_comparisons"],
                    },
                    f,
                    indent=2,
                )
            print(f"[ARC] Wrote full answer report: {report_path}")
        except Exception as exc:
            print(f"[ARC] Could not write answer report: {exc}")

    print("\n" + "=" * 80)
    print("ARC-AGI RESULTS (Million-Brains)")
    print("=" * 80)
    print(f"{'Metric':<28} {'Value':>18}")
    print("-" * 48)
    print(f"{'Test cases':<28} {summary['mbr']['tests']:>18}")
    print(f"{'Parsed outputs':<28} {summary['mbr']['parsed']:>18}")
    print(f"{'Exact matches':<28} {summary['mbr']['correct']:>18}")
    print(f"{'Accuracy':<28} {summary['mbr']['accuracy']:>17.2%}")
    print(f"{'Parse rate':<28} {summary['mbr']['parse_rate']:>17.2%}")
    print(f"{'Total elapsed (s)':<28} {summary['mbr']['time']:>18.2f}")
    print(
        f"{'Avg elapsed / test (s)':<28} "
        f"{summary['mbr']['time'] / max(1, summary['mbr']['tests']):>18.2f}"
    )
    print(f"{'Tokens / sec':<28} {summary['mbr']['tps']:>18.2f}")
    print("=" * 80 + "\n")
    return summary


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Million-Brains DiffusionGemma benchmark and ARC-AGI evaluation"
    )
    parser.add_argument(
        "--arc-profile",
        choices=("local", "kaggle", "auto", "off"),
        default=ARC_DATA_PROFILE,
        help="ARC data source: local data/, Kaggle competition input, auto-detect, or off",
    )
    parser.add_argument(
        "--arc-split",
        choices=tuple(ARC_SPLIT_FILES.keys()),
        default=ARC_DATA_SPLIT,
        help="ARC split to score (training or evaluation)",
    )
    parser.add_argument(
        "--eval-challenges",
        default=EVAL_CHALLENGES_PATH,
        help="Override ARC challenges JSON path",
    )
    parser.add_argument(
        "--eval-solutions",
        default=EVAL_SOLUTIONS_PATH,
        help="Override ARC solutions JSON path",
    )
    parser.add_argument(
        "--eval-max-tasks",
        type=int,
        default=EVAL_MAX_TASKS,
        help="Limit number of ARC tasks (default: all)",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
        help="Per-task generation budget for ARC evaluation",
    )
    parser.add_argument(
        "--run-demo-benchmark",
        action="store_true",
        help="Also run the built-in single-prompt demo benchmark",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Run only the demo benchmark (ignore ARC paths even if set)",
    )
    parser.add_argument(
        "--no-arc-visuals",
        action="store_true",
        help="Disable per-challenge visual grade cards and PNG exports",
    )
    # parse_known_args: Jupyter/Kaggle/Colab inject `-f <kernel.json>` into sys.argv
    args, _unknown = parser.parse_known_args(argv)
    challenges_path, solutions_path, source = resolve_arc_eval_paths(
        profile=args.arc_profile,
        split=args.arc_split,
        challenges_override=args.eval_challenges,
        solutions_override=args.eval_solutions,
    )
    args.eval_challenges = challenges_path
    args.eval_solutions = solutions_path
    args.arc_source = source
    return args


# =============================================================================
# BENCHMARK HARNESS
# =============================================================================
def benchmark(
    vllm_llm: LLM,
    tokenizer: Any,
    prompt: str,
    max_new: int = TARGET_MAX_TOKENS,
):
    """Run Million-Brains DiffusionGemma conditioned denoising benchmark."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: MILLION-BRAINS-DIFFUSIONGEMMA (BENCHMARK_K={K} — demo only, not ARC eval)")
    print("=" * 80)
    print(f"Prompt (first 180 chars): {prompt[:180]}...")
    print(
        f"Target generation length: {max_new} tokens | Block size: {BLOCK_SIZE} | K: {K}"
    )
    print("-" * 80)

    print("\n[MBR] Running Million-Brains DiffusionGemma conditioned denoising ...")
    t0 = time.perf_counter()
    allocator = PermutationFeatureSlotAllocator(
        internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=K
    )
    mbr_res = million_brains_dflash_generate(
        vllm_llm,
        tokenizer,
        prompt,
        max_new_tokens=max_new,
        k=K,
        block_size=BLOCK_SIZE,
        allocator=allocator,
        enable_reallocation=ENABLE_FEATURE_REALLOCATION,
        seed=SEED,
    )
    t_mbr = time.perf_counter() - t0
    mbr_tps = mbr_res["num_tokens"] / max(1e-6, t_mbr)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Value':>22}")
    print("-" * 54)
    print(f"{'Generated tokens':<30} {mbr_res['num_tokens']:>22}")
    print(f"{'Wall time (s)':<30} {t_mbr:>22.2f}")
    print(f"{'Tokens / sec':<30} {mbr_tps:>22.2f}")
    print(f"{'Super-blocks executed':<30} {mbr_res['num_superblocks']:>22}")
    print(
        f"{'Avg accepted tokens / block':<30} {mbr_res['avg_accepted_per_block']:>22.2f}"
    )
    print(
        f"{'Feature reallocations':<30} {mbr_res.get('feature_reallocations', 0):>22}"
    )
    print(f"{'Divergence events':<30} {mbr_res['reframe_events']:>22}")
    if mbr_res.get("circuit_smoothing_enabled"):
        print(
            f"{'Avg circuit blend λ':<30} {mbr_res.get('avg_blend_lambda', 1.0):>22.3f}"
        )

    print("\n--- Sample output ---")
    print(
        mbr_res["final_text"][-600:]
        if len(mbr_res["final_text"]) > 600
        else mbr_res["final_text"]
    )

    print("\n[MBR] Feature allocation history (last 6 super-blocks):")
    for i, feats in enumerate(mbr_res["feature_history"][-6:]):
        print(f"    SB {len(mbr_res['feature_history']) - 6 + i:02d}: {feats}")

    print("\n[MBR] Acceptance trajectory (per super-block):")
    print("   ", [round(a, 3) for a in mbr_res["acceptance_history"][-12:]])

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80 + "\n")

    return {"mbr": {**mbr_res, "tps": mbr_tps, "time": t_mbr}}


# =============================================================================
# MAIN ENTRY POINT (Kaggle script style - just run the file)
# =============================================================================
if __name__ == "__main__":
    args = parse_cli_args()

    print(
        "\n[million_brains_dflash.py] Starting Million-Brains DiffusionGemma Kaggle run"
    )
    print(f"    SCRIPT_VERSION={SCRIPT_VERSION}")
    print(
        f"    ARC_HYPOTHESIS_SLOTS={ARC_HYPOTHESIS_SLOTS} (Phase-1 proposal pool) | "
        f"BENCHMARK_K={K} (demo only) | BLOCK_SIZE={BLOCK_SIZE}"
    )
    print(
        f"    ENGINE: DiffusionGemma canvas={DIFFUSION_CANVAS_LENGTH} denoise_k={K} | "
        f"spatial_ensemble={ARC_SPATIAL_GRID_ENSEMBLE} "
        f"phase1_parallel={ARC_PHASE1_PROMPT_PARALLELISM}"
    )
    print(f"    SEED={SEED}, TARGET_MAX_TOKENS={TARGET_MAX_TOKENS}")
    print_arc_data_config(
        args.eval_challenges, args.eval_solutions, args.arc_source
    )

    # Optional one-time prefetch when Kaggle Internet is enabled
    ensure_model_available()

    model_name = "unknown"
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

    # 4) Sanity: force the banner again so it is unmistakable in the log
    print_one_million_brains_banner(True)

    verify_inference_engine(vllm_llm, tokenizer)

    if run_arc_eval:
        print_post_load_arc_config()

    # 5) Evaluation / benchmark
    results: Dict[str, Any] = {}
    run_demo = args.demo_only or args.run_demo_benchmark or not run_arc_eval

    if run_arc_eval:
        results["arc"] = evaluate_arc_dataset(
            vllm_llm,
            tokenizer,
            args.eval_challenges,
            args.eval_solutions,
            max_tasks=args.eval_max_tasks,
            max_new_tokens=args.eval_max_new_tokens,
            split=args.arc_split,
            visual_grading=ARC_VISUAL_GRADING and not args.no_arc_visuals,
        )

    if run_demo:
        results["demo"] = benchmark(
            vllm_llm, tokenizer, BENCHMARK_PROMPT, max_new=TARGET_MAX_TOKENS
        )


    # 6) Final summary line (useful when scanning Kaggle logs)
    if "demo" in results:
        demo = results["demo"]
        print(
            "[FINAL][demo] MILLION-BRAINS TPS: %.2f | reallocs: %d | Avg accept: %.2f"
            % (
                demo["mbr"]["tps"],
                demo["mbr"].get("feature_reallocations", 0),
                demo["mbr"]["avg_accepted_per_block"],
            )
        )
    if "arc" in results:
        arc = results["arc"]
        print(
            "[FINAL][arc] MBR acc: %.2f%% | tests: %d"
            % (
                arc["mbr"]["accuracy"] * 100,
                arc["mbr"]["tests"],
            )
        )

    # Optional: write a small artifact so the Kaggle "Output" pane has something
    artifact_path = (
        "/kaggle/working/million_brains_dflash_results.json"
        if _on_kaggle()
        else "million_brains_dflash_results.json"
    )
    try:
        payload: Dict[str, Any] = {
            "model": model_name,
            "script_version": SCRIPT_VERSION,
            "benchmark_k": K,
            "arc_hypothesis_slots": ARC_HYPOTHESIS_SLOTS,
            "block_size": BLOCK_SIZE,
        }
        if "demo" in results:
            payload["demo"] = {
                k: v
                for k, v in results["demo"]["mbr"].items()
                if k != "final_text"
            }
        if "arc" in results:
            arc_out = dict(results["arc"])
            arc_out.pop("per_task", None)
            payload["arc"] = arc_out
            payload["eval_challenges"] = args.eval_challenges
            payload["eval_solutions"] = args.eval_solutions

        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[ARTIFACT] Wrote {artifact_path}")
    except Exception:
        pass

    print(
        "\n[million_brains_dflash.py] All done. You can now inspect the generated samples and the metrics above."
    )
