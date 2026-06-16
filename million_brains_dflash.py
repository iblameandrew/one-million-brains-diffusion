#!/usr/bin/env python3
"""
million_brains_dflash.py - one-million-brains-dflash: Permutation-Gated Feature-Slot Allocator + 12 Personality Features
              + Token-Level Cross-Stream Integration + Adaptive Feature Reallocation
              (the Fast Million Brains approach)

This is a complete, self-contained, heavily commented Kaggle script.
It installs vLLM, preemptively live-edits (monkey-patches + file fallback) the DFlash / vLLM draft mechanisms,
injects the full one-million-brains-dflash combinatorial architecture (honoring the Fast Million Brains approach),
then runs a rigorous benchmark comparing classic (single-path Para-DFlash style) vs. full one-million-brains-dflash
(K=4 dynamic feature-slot allocation).

===============================================================================
MATHEMATICAL & ARCHITECTURAL OVERVIEW (read this first - educational)
===============================================================================

Base Para-DFlash Core (generalized speculative / block-parallel drafting):
- Super-block size M = K * block_size. One "super-block" = K parallel reasoning streams,
  each stream of length `block_size`.
- Hierarchical attention inside drafter:
    * Full (non-causal / bidirectional) attention permitted inside each stream's block.
    * Causal / ancestor attention between blocks and across the main prefix.
- One drafter forward (or batched equivalent) produces the entire horizon of M candidate tokens.
- One target forward on the long candidate sequence(s) produces verification logits / logprobs.
- Generalized cumprod acceptance: for each parallel candidate we walk left-to-right,
  at position t compute accept_p = min(1, target_p(token_t) / draft_p(token_t)).
  Draw u ~ Uniform(0,1). Accept while u < accept_p, else reject-and-stop (or resample).
  We track a "rolling target hidden" conceptually and advance the main KV offset by the
  number of accepted tokens (variable per super-block). This is exactly the multi-step
  speculative acceptance generalized across K branches.

one-million-brains-dflash Permutation-Gated Feature-Slot Allocator (Fast Million Brains approach):
- This directly implements the Fast Million Brains approach: a compact bank of 12 high-signal
  personality features is combinatorially permuted at every super-block into K parallel
  "brains" (feature-slots). By rapidly re-allocating different thinking styles across many
  simultaneous reasoning streams, we obtain the robustness, creativity, and error-resistance
  of a million specialized reasoners while keeping the cost close to a single batched drafter
  forward pass per horizon.
- Fixed bank of 12 Personality Features (each feature is a direction in representation space
  with associated sampling biases):
    ["PreciseAnchor", "CreativeExplorer", "LogicalReasoner", "SelfCritic", "Reframer",
     "Synthesizer", "DevilAdvocate", "PatternMatcher", "EdgeCaseHunter", "ContextGrounding",
     "Abstractor", "MirrorReflector"]
- Allocator: lightweight hash gate (plus optional tiny MLP) that ingests a pooled hidden state
  from the *previous verification step*. It produces a deterministic permutation (ordered selection
  without replacement) of K distinct features out of the 12 via combinatorial unranking over
  the P(12, K) possible injections.
- The K selected features are assigned to K **feature-slots**. Each parallel drafter stream
  receives exactly one active feature for that super-block.
- Injection points inside each super-block:
    * feature_emb = nn.Embedding(12, hidden_size)       # fixed personality feature vectors
    * segment_emb + stream_pos_emb (slot_id, local_pos) # Stream-specific positional offsets
    * hidden = token_emb + segment_emb + feature_emb[feature_slot_ids]
    * Per-feature gating vectors (elementwise scale/bias applied after cross-stream integration)
    * Feature-specific temperature / top_p used during the block proposal phase for that slot.
- Selection is a pure function of the hash of the pooled verification state => the exact same
  history always yields the exact same sequence of feature-slot allocations.

Token-Level Cross-Stream Integration & Native Parallelism:
- Two-phase proposal strategy (educational - high-level sampling equivalent here):
    Phase 1 (Independent Stream Draft): K streams generate their blocks using their currently
           slotted personality feature (different feature embeddings + feature-specific
           sampling hyperparameters).
    Phase 2 (Cross-Stream Integration): After raw proposals exist, a lightweight fusion step
           combines signals across streams (priority to "Synthesizer" feature if active,
           otherwise token-wise majority / best under target). In a true kernel patch this
           phase would run a second forward pass under a custom attention mask permitting
           controlled information flow between the K feature-slotted streams before the
           target verification.
- Stream-specific positional offsets keep the K parallel streams distinguishable even when
  sharing the same underlying weights.

Circuit Transition Smoothing Block (CTSB, inter-super-block):
- Discourse State Buffer (DSB): slow EMA of verification pooled state + step structure.
- Geodesic slot step: at most CTSB_MAX_SLOT_SWAPS feature changes per super-block.
- SPI/CBF: interpolate sampling params and feature embeddings toward the new circuit.
- TAFK: stream-level commit selection with discourse coherence + style-jump penalties.

Adaptive Feature Reallocation (Inter-Block):
- After target verification we record per-slot acceptance rate (accepted_tokens / block_size)
  for the K currently allocated features.
- Reallocation: if a feature-slot's exponential-moving-average acceptance falls below threshold,
  the allocator replaces the feature in that slot by drawing a fresh unused feature from the
  remaining (12-K) pool on the next super-block. This is combinatorial adaptation driven by
  measured utility of each personality feature in context.
- Full super-block rejection: if zero tokens are accepted from the best path, temporarily boost
  proposal diversity (higher temperatures) on the next block and record the event. In a low-level
  implementation a control embedding could be added to force a different trajectory.

Permutation Hashing (rigorous core):
- Let n = NUM_PERSONALITY_FEATURES = 12, k = K = 4.
- Number of possible ordered allocations = P(n, k) = n! / (n-k)! = 12*11*10*9 = 11880.
- A 64-bit (or float-derived) seed s obtained from the pooled hidden state of the prior
  verification step is mapped to a unique injection:
      rank = s % P(n, k)
  The rank is decoded via the factorial number system into an ordered K-tuple of distinct
  feature indices. These indices are assigned to feature-slots 0..K-1 for the current super-block.
- Because the mapping is deterministic and bijective (within the modulus), every possible
  stable allocation of features to slots is reachable, the sequence of allocations is fully
  determined by model state history, and no external randomness is required for the combinatorial
  choice itself (only for the token sampling that happens inside the active features).

Kaggle rules followed strictly:
- Starts with !pip install -q vllm transformers accelerate flash-attn --upgrade
- Immediate robust LIVE EDIT section (file overwrite fallback + runtime sys.modules + subclass)
- Prints the exact " ONE-MILLION-BRAINS-FLASH INITIALIZED " banner.
- Easy toggles at the very top after shebang/imports.
- Ends with full head-to-head benchmark: tokens/sec, avg accepted tokens, feature-slot reallocation
  count, allocator decisions, and side-by-side generated samples.

Ready to paste into a Kaggle Python script kernel or notebook cell block.
GPU: T4 / L4 / A10 / A100 all work (uses 1.5B-3B class models by default for headroom).
"""

# =============================================================================
# KAGGLE INSTALL - must be the first executable line when pasted into notebook
# The literal "!pip ..." line below is REQUIRED by the spec for notebook paste.
# When this .py is executed directly (plain python / Kaggle script kernel),
# the guard below performs the equivalent install so the file stays self-contained.
# =============================================================================
# !pip install -q "transformers>=4.57" safetensors accelerate vllm kagglehub huggingface_hub --upgrade
try:
    import IPython

    _in_notebook = IPython.get_ipython() is not None
except Exception:
    _in_notebook = False

if not _in_notebook:
    # Direct execution path (plain .py or Kaggle "Script" kernel): do real install
    import subprocess, sys

    try:
        print("[INSTALL] Running pip via subprocess (plain-Python execution path)...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "transformers>=4.57",
                "safetensors",
                "accelerate",
                "vllm",
                "--upgrade",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as _inst_e:
        print(
            "[INSTALL] Subprocess pip skipped/failed (packages may already exist):",
            _inst_e,
        )
else:
    # Notebook path: pip is usually run via the !pip magic line, but also try subprocess so
    # kagglehub is available for Kaggle-native model download when huggingface.co DNS fails.
    import subprocess, sys

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "transformers>=4.57",
                "safetensors",
                "kagglehub",
                "huggingface_hub",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

# =============================================================================
# TOGGLES - ALL USER CONTROLS LIVE HERE (edit and re-run)
# =============================================================================
SCRIPT_VERSION = "2026-06-16q"  # bump when re-uploading to Kaggle to confirm latest script
K = 4  # number of parallel drafter streams / feature-slots per super-block
NUM_PERSONALITY_FEATURES = 12  # size of the fixed personality feature bank; do not change unless you extend the list below
BLOCK_SIZE = 6  # tokens each stream proposes per super-block (M = K * BLOCK_SIZE)
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
# Kaggle notebook input root (Add Input -> Notebooks -> godelcomplete/qwen3-5-4b-dflash)
KAGGLE_QWEN_BUNDLE_ROOT = "/kaggle/input/notebooks/godelcomplete/qwen3-5-4b-dflash"
LOCAL_MODEL_PATH = KAGGLE_QWEN_BUNDLE_ROOT
LOCAL_BASE_DIR = f"{KAGGLE_QWEN_BUNDLE_ROOT}/Qwen3.5-4B"  # BASE Qwen — generation target (NOT draft)
LOCAL_DFLASH_DIR = f"{KAGGLE_QWEN_BUNDLE_ROOT}/Qwen3.5-4B-DFlash"  # DFlash draft only
BASE_BUNDLE_DIR_NAMES = frozenset({"qwen3.5-4b", "qwen3-5-4b"})
DRAFT_BUNDLE_DIR_NAMES = frozenset({"qwen3.5-4b-dflash", "qwen3-5-4b-dflash"})
# vLLM load safety for custom DFlash / Qwen3.5 checkpoints (avoid pooling + torch.compile crash)
VLLM_ENFORCE_EAGER = True  # required for custom architectures on vLLM v1
VLLM_RUNNER = "generate"  # do not let vLLM pick pooling/embedding runner
VLLM_FALLBACK_TO_HF = True  # HuggingFace wrapper if vLLM still cannot load the checkpoint
PREFER_HF_INFERENCE = True  # Qwen3.5-4B is multimodal; HF is more reliable than vLLM on Kaggle T4
SKIP_VLLM_FOR_QWEN35 = True  # skip vLLM attempts for qwen3_5 / ConditionalGeneration checkpoints
AUTO_PREFETCH_TO_WORKING = True  # when online, cache a small Qwen checkpoint into /kaggle/working
PREFETCH_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # safe default for T4/L4 (22 GB)
KAGGLEHUB_MODEL_HANDLE = "qwen-lm/qwen2.5/transformers/1.5b-instruct"  # Kaggle Models (no huggingface.co DNS needed)
KAGGLE_DATASET_HANDLE = "ragnar123/qwen2-5-1-5b"  # optional dataset fallback on Kaggle
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
EVAL_MAX_NEW_TOKENS = 512  # per-task generation budget for ARC prompts
ARC_VISUAL_GRADING = True  # print + display a grade card for every test case
ARC_PRINT_ALL_ANSWERS = True  # print every prediction vs ground-truth JSON for every test case
ARC_EVAL_VERBOSE = False  # False = hide MBR realloc/super-block spam during ARC eval
# ARC generation: "direct" = one greedy/sampled generate (works on HF). "speculative" = classic dflash loop.
# Speculative with the SAME model as draft+target + draft temp 0.7 vs greedy verify → mass rejection + garbage.
ARC_EVAL_GENERATION_MODE = "direct"
ARC_USE_CHAT_TEMPLATE = True  # Qwen3.5 expects chat_template.jinja wrapping
ARC_DISABLE_THINKING = True  # Qwen3.5: skip <think> blocks that eat token budget
ARC_ASSISTANT_PREFILL = "[["  # chat prefill anchors generation as a JSON grid
ARC_GENERATION_TEMPERATURE = 0.0  # greedy JSON for grid tasks
ARC_RUN_MBR_EVAL = True  # False = skip slow million-brains pass (classic-only smoke tests)
ARC_SAVE_GRADE_IMAGES = True  # save PNG grade cards (arc_grades/ or /kaggle/working/arc_grades/)
ARC_SHOW_TRAIN_EXAMPLES = True  # include train pair thumbnails on each grade card
ARC_ANSWER_REPORT_PATH = None  # None = auto (/kaggle/working/arc_answer_report.json or arc_answer_report.json)

# =============================================================================
# STANDARD LIBRARY + ML IMPORTS
# =============================================================================
import os
import sys
import socket
import argparse
import re
import math
import time
import random
import json
import hashlib
import traceback
import re
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Heavy libraries (installed above)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams


def _resolve_vllm_draft_module():
    """
    Locate vLLM's draft/spec-decode module across API versions.

    Legacy vLLM (<=0.6): vllm.spec_decode.draft_model (DraftModel)
    Current vLLM (v1 engine): vllm.v1.spec_decode.draft_model (DraftModelProposer)
    """
    import importlib
    import types

    candidates = (
        "vllm.spec_decode.draft_model",
        "vllm.v1.spec_decode.draft_model",
        "vllm.v1.spec_decode",
    )
    for mod_path in candidates:
        try:
            return importlib.import_module(mod_path)
        except ModuleNotFoundError:
            continue

    print(
        "[IMPORT] vLLM draft module not found (tried "
        + ", ".join(candidates)
        + "); using in-process stub for live-edit fallback."
    )
    stub = types.ModuleType("vllm_draft_stub")
    sys.modules.setdefault("vllm.spec_decode.draft_model", stub)
    return stub


vllm_draft_module = _resolve_vllm_draft_module()  # for live patching target

# =============================================================================
# FIXED BANK OF 12 PERSONALITY FEATURES
# Each feature is a distinct direction that can be injected into a feature-slot.
# The PermutationFeatureSlotAllocator selects K distinct features via a deterministic
# permutation and assigns them to the K parallel streams for the current super-block.
# =============================================================================
PERSONALITY_FEATURES: List[str] = [
    "PreciseAnchor",  # low temperature, anchors strongly to given facts and constraints
    "CreativeExplorer",  # higher temperature / broad top_p, favors novelty and less-trodden paths
    "LogicalReasoner",  # medium temperature with bias toward explicit step-by-step chains
    "SelfCritic",  # tends to surface contradictions and reduce over-commitment
    "Reframer",  # biases toward re-interpreting the problem statement or constraints
    "Synthesizer",  # prefers combinations; receives priority during cross-stream fusion
    "DevilAdvocate",  # stress-tests the currently favored line of reasoning
    "PatternMatcher",  # strongly attends to numerical, structural, and repetition patterns
    "EdgeCaseHunter",  # deliberately explores boundary conditions and low-probability inputs
    "ContextGrounding",  # pulls in broader real-world constraints and background knowledge
    "Abstractor",  # lifts specifics into higher-level rules or invariants
    "MirrorReflector",  # meta-feature that conditions on the behavior of the other active features
]

# Feature-specific generation hyper-parameters used during the proposal phase for the
# stream that has been allocated that personality feature. This is the high-level
# equivalent of "feature-specific temperature/top_p + post-attention gating".
FEATURE_PARAMS: Dict[str, Dict[str, float]] = {
    "PreciseAnchor": {"temperature": 0.35, "top_p": 0.82, "repetition_penalty": 1.08},
    "CreativeExplorer": {
        "temperature": 1.15,
        "top_p": 0.98,
        "repetition_penalty": 1.00,
    },
    "LogicalReasoner": {"temperature": 0.55, "top_p": 0.90, "repetition_penalty": 1.05},
    "SelfCritic": {"temperature": 0.65, "top_p": 0.88, "repetition_penalty": 1.12},
    "Reframer": {"temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.02},
    "Synthesizer": {"temperature": 0.60, "top_p": 0.93, "repetition_penalty": 1.03},
    "DevilAdvocate": {"temperature": 0.80, "top_p": 0.91, "repetition_penalty": 1.06},
    "PatternMatcher": {"temperature": 0.45, "top_p": 0.85, "repetition_penalty": 1.04},
    "EdgeCaseHunter": {"temperature": 0.75, "top_p": 0.94, "repetition_penalty": 1.07},
    "ContextGrounding": {
        "temperature": 0.50,
        "top_p": 0.89,
        "repetition_penalty": 1.01,
    },
    "Abstractor": {"temperature": 0.70, "top_p": 0.96, "repetition_penalty": 1.05},
    "MirrorReflector": {"temperature": 0.85, "top_p": 0.87, "repetition_penalty": 1.09},
}

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
    In a real kernel patch (inside DFlashDraftModel or a custom attention forward)
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
# CUMPROD ACCEPTANCE (core speculative primitive, generalized to K paths)
# =============================================================================
def compute_accepted_tokens(
    draft_token_ids: List[int],
    target_logprobs_for_exact_tokens: List[float],
    draft_logprobs_for_exact_tokens: List[float],
    min_accept: float = 1e-6,
) -> Tuple[List[int], float]:
    """
    Walk the proposed block left-to-right performing the classic speculative test.

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
            # In full speculative we would draw a new token from (target - draft) here.
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
# ONE-MILLION-BRAINS-FLASH GENERATE (the full algorithm)
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
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full one-million-brains-dflash loop (permutation-driven feature-slot allocation).

    - Uses vLLM for fast batched proposal across K feature-slotted streams
      (the "drafter forward" for the entire super-block horizon).
    - Uses vLLM prompt_logprobs for the target verification forward on the K candidate sequences.
    - The PermutationFeatureSlotAllocator selects a fresh K-permutation of personality features
      for every super-block based on the pooled state of the previous verification.
    - CircuitTransitionSmoother (CTSB) interpolates circuit handoffs for semantic coherence.
    - Adaptive reallocation: under-performing feature-slots have their personality feature
      replaced from the unused pool on subsequent blocks.
    - This realizes the Fast Million Brains approach at inference time.
    - Returns rich stats + final text.
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

    # State
    current_text = prompt
    generated_ids: List[int] = []
    total_accepted = 0
    total_blocks = 0
    feature_reallocations = 0
    acceptance_history: List[float] = []
    feature_history: List[List[str]] = []
    reframe_events = 0
    prev_acceptance_rate = 0.5
    prev_accepted_len = 0

    # Per-slot EMA acceptance (used to decide when to re-allocate a different personality feature)
    ema_accept = [0.5] * k
    # Bootstrap: initial allocation is simply the first K features (will be overridden by the allocator immediately)
    active_feature_indices: List[int] = list(range(k))

    max_blocks = (max_new_tokens // max(1, block_size)) + 4

    for sb in range(max_blocks):
        if len(generated_ids) >= max_new_tokens:
            break
        total_blocks += 1

        # 1) Allocator decision (permutation of personality features into feature-slots)
        pooled = make_pooled_state(generated_ids, sb, tokenizer=tokenizer)
        alloc_out = allocator(pooled, sb)
        target_feature_indices = alloc_out["feature_indices"]
        target_feature_names = alloc_out["feature_names"]
        feature_history.append(target_feature_names)

        # 1b) CTSB: geodesic slot step + SPI/CBF smoothing before drafting
        blend_lambda = 1.0
        if smoother is not None:
            full_rejection = prev_accepted_len == 0 and sb > 0
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

        # 2) Independent Draft phase (K parallel proposals via vLLM batch)
        #    One batched "drafter forward" for the whole super-block horizon, each stream
        #    conditioned on its currently allocated personality feature.
        draft_sampling = []
        for i in range(k):
            p = feature_params[i]
            sp = SamplingParams(
                temperature=p["temperature"]
                + (
                    REFRAME_TEMP_BOOST
                    if reframe_events > 0 and sb == total_blocks - 1
                    else 0.0
                ),
                top_p=p["top_p"],
                max_tokens=block_size,
                logprobs=1,
                stop_token_ids=None,
            )
            draft_sampling.append(sp)

        # vLLM handles the heterogeneous batched drafting efficiently.
        outs = vllm_llm.generate([current_text] * k, draft_sampling)
        proposals: List[List[int]] = []
        draft_lps: List[List[float]] = []
        for out in outs:
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

        # 3) Cross-stream integration / fusion phase (the high-level "Group Think")
        #    CTSB enabled: anchor-stream fused candidate (coherent, no per-token splicing).
        #    Legacy: Synthesizer priority or position-wise majority vote.
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

        # 4) Target verification forward on the K candidate sequences (+ fused)
        #    Single batched call giving us the target's view of every drafted token.
        candidates_for_verify = []
        base = current_text
        for pr in proposals:
            append_text = tokenizer.decode(pr, skip_special_tokens=True)
            candidates_for_verify.append(base + append_text)
        fused_text = tokenizer.decode(fused_proposal, skip_special_tokens=True)
        candidates_for_verify.append(base + fused_text)

        verify_params = make_target_verify_sampling_params()
        verify_outs = vllm_llm.generate(candidates_for_verify, verify_params)

        # Extract target logprobs for the drafted region of each candidate
        target_lps_per_path: List[List[float]] = []
        for j, vout in enumerate(verify_outs):
            target_ids = proposals[j] if j < len(proposals) else fused_proposal
            drafted_lps = extract_target_logprobs_for_draft(
                vout.prompt_logprobs, target_ids
            )
            target_lps_per_path.append(drafted_lps)

        # 5) Run generalized cumprod acceptance on every path
        best_accepted: List[int] = []
        best_rate = -1.0
        path_rates: List[float] = []
        accepted_per_path: List[List[int]] = []

        for j in range(k):
            acc, rate = compute_accepted_tokens(
                proposals[j],
                target_lps_per_path[j],
                draft_lps[j],
            )
            path_rates.append(rate)
            accepted_per_path.append(acc)
            if smoother is None and rate > best_rate:
                best_rate = rate
                best_accepted = acc

        # Also test the fused candidate
        acc_f, rate_f = compute_accepted_tokens(
            fused_proposal,
            target_lps_per_path[-1],
            draft_lps[0],
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

        # 6) Commit accepted tokens, update main sequence + rolling offset
        accepted_len = len(best_accepted)
        if accepted_len > 0:
            new_ids = best_accepted
            generated_ids.extend(new_ids)
            append_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            current_text = current_text + append_text
            total_accepted += accepted_len
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

        # 7) Adaptive feature reallocation (the "Mirror" mechanism, stripped of astrology)
        if enable_reallocation:
            for i in range(k):
                ema_accept[i] = 0.7 * ema_accept[i] + 0.3 * path_rates[i]

            for i in range(k):
                if ema_accept[i] < ACCEPTANCE_THRESHOLD:
                    # Draw a replacement from the currently unused personality features.
                    unused = [
                        r
                        for r in range(NUM_PERSONALITY_FEATURES)
                        if r not in active_feature_indices
                    ]
                    if unused:
                        new_feat = unused[(sb * 31 + i) % len(unused)]
                        old_name = PERSONALITY_FEATURES[active_feature_indices[i]]
                        active_feature_indices[i] = new_feat
                        feature_reallocations += 1
                        ema_accept[i] = 0.55
                        if verbose:
                            print(
                                f"  [REALLOC] Slot {i} feature {old_name} -> "
                                f"{PERSONALITY_FEATURES[new_feat]} (EMA accept {ema_accept[i]:.3f})"
                            )

        # 8) Full rejection handling (equivalent to "Reframe")
        if accepted_len == 0 and enable_reallocation:
            reframe_events += 1
            if verbose:
                print(
                    f"  [DIVERGENCE] Super-block {sb} produced zero accepted tokens. "
                    "Boosting proposal diversity for next block."
                )

        # 9) Diagnostic logging (feature-slot view)
        if verbose:
            gmask = GroupThinkMask(
                k=k,
                block_size=block_size,
                phase="integration" if accepted_len > 0 else "draft",
            )
            if sb < 2 or accepted_len == 0:
                smooth_note = (
                    f" | λ={blend_lambda:.2f}"
                    if smoother is not None
                    else ""
                )
                print(
                    f"    Super-block {sb:02d} | features={active_feature_names} | "
                    f"accepted={accepted_len}/{block_size}{smooth_note} | mask={gmask.describe()}"
                )

        if len(generated_ids) >= max_new_tokens:
            break

    final_text = current_text
    tokens_per_block = total_accepted / max(1, total_blocks)

    return {
        "final_text": final_text,
        "generated_ids": generated_ids,
        "num_tokens": len(generated_ids),
        "num_superblocks": total_blocks,
        "total_accepted": total_accepted,
        "avg_accepted_per_block": tokens_per_block,
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
    }


# =============================================================================
# CLASSIC PARA-DFLASH (K=1, no orchestrator, plain cumprod speculative)
# =============================================================================
def classic_dflash_generate(
    vllm_llm: LLM,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    block_size: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Baseline single-path Para-DFlash style speculative loop (educational K=1).
    One drafter proposal of `block_size` tokens, one target verify, cumprod accept, repeat.
    """
    random.seed(seed)
    current_text = prompt
    generated_ids: List[int] = []
    total_accepted = 0
    total_blocks = 0
    acceptance_history: List[float] = []

    max_blocks = (max_new_tokens // max(1, block_size)) + 4

    for sb in range(max_blocks):
        if len(generated_ids) >= max_new_tokens:
            break
        total_blocks += 1

        # Drafter forward (single path)
        sp = SamplingParams(
            temperature=BASE_TEMPERATURE,
            top_p=BASE_TOP_P,
            max_tokens=block_size,
            logprobs=1,
        )
        out = vllm_llm.generate([current_text], sp)[0]
        draft_ids = list(out.outputs[0].token_ids)[:block_size]
        draft_lps = []
        if out.outputs[0].logprobs:
            for tid in draft_ids:
                draft_lps.append(
                    extract_logprob_for_token(out.outputs[0].logprobs, tid)
                )
        else:
            draft_lps = [-0.75] * len(draft_ids)

        # Target verification forward on the single extended candidate
        candidate = current_text + tokenizer.decode(draft_ids, skip_special_tokens=True)
        vout = vllm_llm.generate(
            [candidate], make_target_verify_sampling_params()
        )[0]
        target_lps = extract_target_logprobs_for_draft(
            vout.prompt_logprobs, draft_ids
        )

        acc, rate = compute_accepted_tokens(draft_ids, target_lps, draft_lps)
        acceptance_history.append(rate)

        if acc:
            generated_ids.extend(acc)
            current_text = current_text + tokenizer.decode(
                acc, skip_special_tokens=True
            )
            total_accepted += len(acc)

        if len(generated_ids) >= max_new_tokens:
            break

    final_text = current_text
    tokens_per_block = total_accepted / max(1, total_blocks)
    return {
        "final_text": final_text,
        "generated_ids": generated_ids,
        "num_tokens": len(generated_ids),
        "num_superblocks": total_blocks,
        "total_accepted": total_accepted,
        "avg_accepted_per_block": tokens_per_block,
        "feature_reallocations": 0,
        "acceptance_history": acceptance_history,
        "feature_history": [["ClassicDFlash"]] * total_blocks,
        "reframe_events": 0,
    }


# =============================================================================
# ONEMILLIONBRAINSDFLASHDRAFTMODEL (module-level so patcher and file-injection can always see it)
# =============================================================================
class MillionBrainsDFlashDraftModel:
    """
    Stand-in / monkey-patched replacement for DFlashDraftModel (or vLLM's DraftModel).
    In a real kernel patch this would contain the feature_emb tables, the hierarchical
    mask builder for cross-stream integration, the cumprod + KV offset logic, and the
    PermutationFeatureSlotAllocator hook on every super-block boundary.
    The high-level implementation lives in million_brains_dflash_generate().
    """

    def __init__(self, *args, **kwargs):
        self.k = K
        self.block_size = BLOCK_SIZE
        self.enable_reallocation = ENABLE_FEATURE_REALLOCATION
        self.feature_allocator = None  # filled by _live_edit_dflash
        print(
            "[MillionBrainsDFlashDraftModel] Stand-in initialized (full feature-slot allocator injected at runtime)"
        )

    def propose_with_allocator(self, pooled_state, step):
        if self.feature_allocator is not None:
            return self.feature_allocator(pooled_state, step)
        return list(range(min(self.k, NUM_PERSONALITY_FEATURES)))


# =============================================================================
# LIVE EDIT / MONKEY-PATCH SECTION (the "preemptive" DFlash injection)
# This must run immediately after the pip install and before heavy model loading.
# Strategy:
#   1. Attempt to locate vllm.spec_decode / vllm.v1.spec_decode draft_model on disk and surgically
#      overwrite key classes / methods with one-million-brains-dflash versions (file fallback).
#   2. Always perform runtime monkey-patching via subclass + module replacement.
#   3. Inject a global "MILLION_BRAINS_DFLASH" symbol so user code can detect the patch.
#   4. Emit the exact "ONE-MILLION-BRAINS-FLASH INITIALIZED" banner on success.
# =============================================================================
def _live_edit_dflash() -> bool:
    """
    Robust live-edit routine. Returns True if any patch (file or runtime) succeeded.
    Performs the one-million-brains-dflash injection into vLLM's draft mechanisms.
    """
    patched = False
    print(
        "\n[PREEMPTIVE DFLASH LIVE-EDIT] Scanning for DFlashDraftModel / vLLM draft components (one-million-brains-dflash injection)..."
    )

    # --- Step 1: File-level overwrite (fallback when source is writable) ---
    try:
        import vllm

        vllm_dir = os.path.dirname(vllm.__file__)
        candidate_files = []
        for root, _, files in os.walk(vllm_dir):
            for f in files:
                if "draft" in f.lower() and f.endswith(".py"):
                    candidate_files.append(os.path.join(root, f))
                if "spec_decode" in root and f.endswith(".py"):
                    candidate_files.append(os.path.join(root, f))

        # De-duplicate while preserving order
        seen = set()
        candidate_files = [x for x in candidate_files if not (x in seen or seen.add(x))]

        target_file = None
        for cf in candidate_files:
            try:
                with open(cf, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                if "class DraftModel" in content or "DraftModel" in content:
                    target_file = cf
                    break
            except Exception:
                continue

        if target_file:
            print(f"    Found candidate DFlash source: {target_file}")
            with open(target_file, "r", encoding="utf-8", errors="ignore") as fh:
                original = fh.read()

            # We do not brutally rewrite the whole file (fragile across vLLM versions).
            # Instead we append a large one-million-brains-dflash injection comment block + a small
            # "MillionBrainsDFlashDraftModel" subclass at the bottom, then try to swap at runtime.
            injection = f"""
# =============================================================================
# MILLION-BRAINS-DFLASH INJECTION (auto-inserted by million_brains_dflash.py at {time.strftime("%Y-%m-%d %H:%M:%S")})
# This file was live-edited to support the full PermutationFeatureSlotAllocator,
# 12 personality features, cross-stream integration, and adaptive feature reallocation.
# (Fast Million Brains approach)
# The real algorithmic work happens in the Python-level million_brains_dflash_generate.
# The symbols below are here so that vLLM's speculative path can see them.
# =============================================================================

MILLION_BRAINS_DFLASH_PATCHED = True
MILLION_BRAINS_PERSONALITY_FEATURES = {PERSONALITY_FEATURES!r}
MILLION_BRAINS_K = {K}
MILLION_BRAINS_BLOCK_SIZE = {BLOCK_SIZE}

# The real heavy lifting (permutation allocator, cumprod acceptance, feature reallocation)
# lives in the million_brains_dflash.py that you ran.
try:
    import million_brains_dflash as _pf
    PermutationFeatureSlotAllocator = _pf.PermutationFeatureSlotAllocator
    million_brains_dflash_generate = _pf.million_brains_dflash_generate
    classic_dflash_generate = _pf.classic_dflash_generate
except Exception:
    pass

# Local lightweight stand-in
class MillionBrainsDFlashDraftModel:  # type: ignore
    \"\"\"
    File-injected stand-in for DFlashDraftModel.
    The full one-million-brains-dflash (PermutationFeatureSlotAllocator + 12 personality features
    + cross-stream integration + adaptive feature reallocation, the Fast Million Brains
    approach) lives in the calling million_brains_dflash.py. This object carries the
    feature-slot allocation hook.
    \"\"\"
    def __init__(self, *args, **kwargs):
        self.k = {K}
        self.block_size = {BLOCK_SIZE}
        self.enable_reallocation = {ENABLE_FEATURE_REALLOCATION}
        self.feature_allocator = None
        print("[MillionBrainsDFlashDraftModel] File-injected stand-in ready")

    def propose_with_allocator(self, pooled_state, step):
        if self.feature_allocator is not None:
            return self.feature_allocator(pooled_state, step)
        return list(range(min(self.k, 12)))

# End of one-million-brains-dflash injection
"""
            if "MILLION-BRAINS-DFLASH INJECTION" not in original:
                try:
                    with open(target_file, "a", encoding="utf-8") as fh:
                        fh.write("\n" + injection)
                    print(
                        f"    [FILE PATCH] Appended one-million-brains-dflash injection to {target_file}"
                    )
                    patched = True
                except Exception as e:
                    print(
                        f"    [FILE PATCH] Could not append (likely read-only FS): {e}"
                    )
            else:
                patched = True
    except Exception as e:
        print(f"    [FILE SCAN] Skipped or failed: {e}")

    # --- Step 2: Always do runtime monkey-patch (the reliable path) ---
    try:
        # Inject symbols directly into the vllm draft module namespace
        vllm_draft_module.MILLION_BRAINS_DFLASH_PATCHED = True
        vllm_draft_module.MILLION_BRAINS_PERSONALITY_FEATURES = PERSONALITY_FEATURES
        vllm_draft_module.MILLION_BRAINS_K = K
        vllm_draft_module.MILLION_BRAINS_BLOCK_SIZE = BLOCK_SIZE
        vllm_draft_module.PermutationFeatureSlotAllocator = (
            PermutationFeatureSlotAllocator
        )
        vllm_draft_module.million_brains_dflash_generate = (
            million_brains_dflash_generate
        )
        vllm_draft_module.classic_dflash_generate = classic_dflash_generate

        # Subclass replacement (what user code importing DraftModel will see)
        try:
            OriginalDraft = getattr(vllm_draft_module, "DraftModel", None) or getattr(
                vllm_draft_module, "DraftModelProposer", None
            )
            if OriginalDraft is not None:

                class PatchedDraftModel(OriginalDraft):  # type: ignore
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        self.feature_allocator = PermutationFeatureSlotAllocator(
                            internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=K
                        )
                        self.k = K
                        self.block_size = BLOCK_SIZE
                        print(
                            "[PatchedDraftModel] Runtime subclass active - one-million-brains-dflash ready"
                        )

                draft_cls_name = getattr(OriginalDraft, "__name__", "DraftModel")
                if hasattr(vllm_draft_module, "DraftModel"):
                    vllm_draft_module.DraftModel = PatchedDraftModel
                if hasattr(vllm_draft_module, "DraftModelProposer"):
                    vllm_draft_module.DraftModelProposer = PatchedDraftModel
                sys.modules.setdefault("dflash", type(sys)("dflash"))
                sys.modules["dflash"].DraftModel = PatchedDraftModel
                sys.modules["dflash"].MILLION_BRAINS_DFLASH_PATCHED = True
                print(
                    f"    [RUNTIME PATCH] {vllm_draft_module.__name__}.{draft_cls_name} replaced with one-million-brains-dflash feature-slot aware subclass"
                )
            else:
                # No original DraftModel - still expose our allocator
                sys.modules.setdefault("dflash", type(sys)("dflash"))
                ndm = MillionBrainsDFlashDraftModel()
                ndm.feature_allocator = PermutationFeatureSlotAllocator(
                    internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=K
                )
                sys.modules["dflash"].DraftModel = type(ndm)
                sys.modules["dflash"].MillionBrainsDFlashDraftModel = type(ndm)
                print(
                    "    [RUNTIME PATCH] No original DraftModel found - pure MillionBrainsDFlashDraftModel exposed under 'dflash'"
                )
            patched = True
        except Exception as sub_e:
            print(f"    [RUNTIME SUBCLASS] {sub_e}")

    except Exception as rt_e:
        print(f"    [RUNTIME PATCH] Failed: {rt_e}")

    return patched


# Execute the live edit immediately (this is the moment the "DFlash library" is hijacked with one-million-brains-dflash)
_LIVE_PATCH_SUCCESS = _live_edit_dflash()


# =============================================================================
# REQUIRED BANNER (exact string required by the spec)
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
            " ONE-MILLION-BRAINS-FLASH INITIALIZED  |  K=%d  |  FEATURES=%d  |  REALLOCATION=%s"
            % (K, NUM_PERSONALITY_FEATURES, str(ENABLE_FEATURE_REALLOCATION).upper())
        )
        print(
            " Patch status: %s"
            % ("SUCCESS (file+runtime)" if _LIVE_PATCH_SUCCESS else "RUNTIME ONLY")
        )
        print(f" Script version: {SCRIPT_VERSION}")
    else:
        print(
            " ONE-MILLION-BRAINS-FLASH INITIALIZED (DEGRADED - patch encountered errors, pure-Python fallback active)"
        )
    print(
        "================================================================================\n"
    )


print_one_million_brains_banner(_LIVE_PATCH_SUCCESS)


# =============================================================================
# MODEL LOADING WITH FALLBACK (local/offline first, then HuggingFace when online)
# =============================================================================
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
    """When a mount nests BASE + DFlash, prefer the full causal-LM checkpoint."""
    name = os.path.basename(path).lower()
    draft_rank = 1 if is_dflash_draft_checkpoint(path) else 0
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
    """Lower tuple sorts earlier: prefer DFlash-tuned, then larger Qwen, then lexicographic."""
    name = os.path.basename(path).lower()
    score = 0
    if "dflash" in name:
        score -= 100
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
    """Pick the best available on-disk checkpoint without touching the network."""
    explicit = [p for p in (LOCAL_BASE_DIR, LOCAL_DFLASH_DIR, LOCAL_MODEL_PATH) if p]
    resolved_list: List[str] = []
    seen: set = set()
    for path in explicit:
        resolved = resolve_checkpoint_dir(path)
        if resolved and resolved not in seen:
            seen.add(resolved)
            if resolved != os.path.abspath(path):
                print(f"[LOCAL-LOAD] Resolved nested checkpoint: {path} -> {resolved}")
            resolved_list.append(resolved)

    if prefer_generation:
        for resolved in resolved_list:
            if is_full_causal_lm_checkpoint(resolved):
                return resolved
    if resolved_list:
        return resolved_list[0]

    discovered = discover_local_checkpoints()
    if discovered:
        print(f"[LOCAL-LOAD] Auto-discovered {len(discovered)} local checkpoint(s):")
        for p in discovered:
            role = "FULL" if is_full_causal_lm_checkpoint(p) else (
                "DRAFT" if is_dflash_draft_checkpoint(p) else "UNKNOWN"
            )
            print(f"    - {p} [{role}]")
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
    """
    VRAM-aware + fallback logic exactly as requested.
    Order:
      1. z-lab/Qwen3.5-4B-DFlash (the "special" DFlash-tuned checkpoint - will usually 404)
      2. Qwen/Qwen2.5-3B-Instruct or 1.5B (safe on T4/L4)
      3. If massive VRAM: try 7B or 14B (never 27B without explicit quantization in this script)
    """
    gpu_mem_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = props.total_memory / (1024**3)
        print(f"[GPU] Detected {props.name} with {gpu_mem_gb:.1f} GB VRAM")

    if PREFER_LOCAL_MODELS:
        local_path = resolve_local_model_path()
        if local_path:
            print(f"[MODEL] Using local checkpoint (no HuggingFace HEAD requests): {local_path}")
            return local_path, "vllm"

    ensure_model_available()
    local_path = resolve_local_model_path()
    if local_path:
        print(f"[MODEL] Using local checkpoint after prefetch: {local_path}")
        return local_path, "vllm"

    if not any_download_channel_available():
        raise RuntimeError(
            "[MODEL] No local checkpoint found and no download channel is reachable.\n"
            "  Fix (pick one):\n"
            "    1. Kaggle Settings -> Internet -> ON, restart kernel, re-run ALL cells from the top.\n"
            "    2. Add dataset 'ragnar123/qwen2-5-1-5b' (or any HF model dataset) as notebook Input.\n"
            "    3. Set LOCAL_MODEL_PATH = '/kaggle/input/<your-dataset>/<model-folder>' at the top.\n"
            "    4. Run this once with Internet ON:\n"
            "         import kagglehub\n"
            f"         kagglehub.model_download('{KAGGLEHUB_MODEL_HANDLE}', output_dir='/kaggle/working/Qwen2.5-1.5B-Instruct')"
        )

    candidates = [
        "z-lab/Qwen3.5-4B-DFlash",  # the requested fictional/special model
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ]
    if gpu_mem_gb > 38:
        candidates.insert(1, "Qwen/Qwen2.5-7B-Instruct")
    if gpu_mem_gb > 70:
        candidates.insert(1, "Qwen/Qwen2.5-14B-Instruct")

    for name in candidates:
        print(f"[MODEL] Trying to load: {name}")
        try:
            # Quick tokenizer probe is cheap; the real load happens in load_models()
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            print(f"[MODEL] Tokenizer OK for {name}")
            return name, "vllm"
        except Exception as e:
            print(
                f"    ... {name} not available or tokenizer failed ({type(e).__name__}). Trying next..."
            )
    # Absolute last resort (should never happen)
    return "Qwen/Qwen2.5-1.5B-Instruct", "vllm"


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
    """
    Trust Kaggle bundle folder names over weight heuristics.
    Qwen3.5-4B = base Qwen for generation. Qwen3.5-4B-DFlash = draft only.
    """
    if not model_path:
        return None
    name = os.path.basename(os.path.abspath(model_path)).lower()
    if name in DRAFT_BUNDLE_DIR_NAMES or name.endswith("-dflash"):
        return "draft"
    if "dflash" in name and "qwen" in name:
        return "draft"
    if name in BASE_BUNDLE_DIR_NAMES:
        return "base"
    if "qwen" in name and "4b" in name and "dflash" not in name:
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


def is_dflash_draft_checkpoint(model_path: str) -> bool:
    """
    DFlash draft weights ship fc/hidden_norm but not embed_tokens.
    Folder-name rules take priority: Qwen3.5-4B is NEVER a draft.
    """
    if not local_dir_exists(model_path):
        return False

    role = bundle_folder_role(model_path)
    if role == "base":
        return False
    if role == "draft":
        return True

    if is_full_causal_lm_checkpoint(model_path):
        return False

    name_l = os.path.basename(model_path).lower()
    if "dflash" in name_l:
        return True

    keys = _list_checkpoint_weight_keys(model_path)
    if keys:
        has_fc = any(".fc.weight" in k or k.endswith("fc.weight") for k in keys)
        has_hidden_norm = any("hidden_norm" in k for k in keys)
        if has_fc or has_hidden_norm:
            return True

    try:
        cfg = _read_hf_config(model_path)
        archs = [str(a) for a in (cfg.get("architectures") or [])]
        if any("dflash" in a.lower() or "draft" in a.lower() for a in archs):
            return True
    except Exception:
        pass

    return False


def discover_qwen_dflash_bundle_paths() -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve BASE + DFLASH folders for the qwen3.5-4b-dflash Kaggle bundle.
    Checks KAGGLE_QWEN_BUNDLE_ROOT first, then scans /kaggle/input.
    """
    base_candidates: List[str] = []
    draft_candidates: List[str] = []

    bundle_root = KAGGLE_QWEN_BUNDLE_ROOT
    if bundle_root and os.path.isdir(bundle_root):
        base_candidate = os.path.join(bundle_root, "Qwen3.5-4B")
        draft_candidate = os.path.join(bundle_root, "Qwen3.5-4B-DFlash")
        if local_dir_exists(base_candidate):
            base_candidates.append(os.path.abspath(base_candidate))
        if local_dir_exists(draft_candidate):
            draft_candidates.append(os.path.abspath(draft_candidate))

    for raw, bucket in (
        (LOCAL_BASE_DIR, base_candidates),
        (LOCAL_DFLASH_DIR, draft_candidates),
        (LOCAL_MODEL_PATH, base_candidates),
    ):
        if not raw:
            continue
        resolved = resolve_checkpoint_dir(raw)
        if not resolved:
            continue
        if is_dflash_draft_checkpoint(resolved):
            draft_candidates.append(resolved)
        elif is_full_causal_lm_checkpoint(resolved):
            base_candidates.append(resolved)

    if os.path.isdir("/kaggle/input"):
        found: List[str] = []

        def _walk(base: str, depth: int) -> None:
            if depth > 6 or not os.path.isdir(base):
                return
            if local_dir_exists(base):
                found.append(os.path.abspath(base))
                return
            try:
                entries = sorted(os.listdir(base))
            except OSError:
                return
            for entry in entries:
                if entry.startswith("."):
                    continue
                _walk(os.path.join(base, entry), depth + 1)

        _walk("/kaggle/input", 0)
        for path in found:
            name = os.path.basename(path).lower()
            if "dflash" in name and is_dflash_draft_checkpoint(path):
                draft_candidates.append(path)
            elif is_full_causal_lm_checkpoint(path) and "dflash" not in name:
                base_candidates.append(path)

    def _uniq(paths: List[str]) -> List[str]:
        out: List[str] = []
        seen: set = set()
        for p in paths:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    base_candidates = _uniq(base_candidates)
    draft_candidates = _uniq(draft_candidates)

    def _base_rank(path: str) -> Tuple[int, int, str]:
        name = os.path.basename(path).lower()
        score = 0
        if "dflash" in name:
            score += 1000
        if name in ("qwen3.5-4b", "qwen3-5-4b"):
            score -= 50
        if "4b" in name:
            score -= 10
        if "qwen" in name:
            score -= 5
        return (score, -len(name), path)

    def _draft_rank(path: str) -> Tuple[int, str]:
        name = os.path.basename(path).lower()
        score = 0 if "dflash" in name else 1
        return (score, path)

    base = sorted(base_candidates, key=_base_rank)[0] if base_candidates else None
    draft = sorted(draft_candidates, key=_draft_rank)[0] if draft_candidates else None
    return base, draft


def find_paired_dflash_draft(base_path: str) -> Optional[str]:
    """Locate a DFlash draft sibling for a full base checkpoint."""
    candidates: List[str] = []

    explicit = resolve_checkpoint_dir(LOCAL_DFLASH_DIR)
    if explicit and is_dflash_draft_checkpoint(explicit):
        candidates.append(explicit)

    search_roots: List[str] = []
    for root in (base_path, os.path.dirname(base_path), LOCAL_MODEL_PATH):
        if root and os.path.isdir(root) and root not in search_roots:
            search_roots.append(root)

    for search_root in search_roots:
        try:
            for entry in sorted(os.listdir(search_root)):
                if "dflash" not in entry.lower():
                    continue
                resolved = resolve_checkpoint_dir(os.path.join(search_root, entry))
                if resolved and is_dflash_draft_checkpoint(resolved):
                    candidates.append(resolved)
        except OSError:
            pass

    def _draft_priority(path: str) -> Tuple[int, str]:
        name = os.path.basename(path).lower()
        score = 0 if "dflash" in name else 1
        return (score, path)

    ranked = sorted(set(candidates), key=_draft_priority)
    return ranked[0] if ranked else None


def find_paired_base_checkpoint(dflash_path: str) -> Optional[str]:
    """Locate a full causal-LM checkpoint to pair with a DFlash draft directory."""
    candidates: List[str] = []

    explicit = resolve_checkpoint_dir(LOCAL_BASE_DIR)
    if explicit and is_full_causal_lm_checkpoint(explicit):
        candidates.append(explicit)

    parent = os.path.dirname(dflash_path)
    if os.path.isdir(parent):
        try:
            for entry in sorted(os.listdir(parent)):
                sibling = os.path.join(parent, entry)
                resolved = resolve_checkpoint_dir(sibling)
                if (
                    resolved
                    and resolved != dflash_path
                    and is_full_causal_lm_checkpoint(resolved)
                ):
                    candidates.append(resolved)
        except OSError:
            pass

    for discovered in discover_local_checkpoints():
        if discovered != dflash_path and is_full_causal_lm_checkpoint(discovered):
            candidates.append(discovered)

    def _base_priority(path: str) -> Tuple[int, int, str]:
        name = os.path.basename(path).lower()
        score = 0
        if "dflash" in name:
            score += 1000
        if "qwen" in name:
            score -= 10
        if "4b" in name:
            score -= 8
        if "base" in name or "instruct" in name:
            score -= 6
        return (score, -len(name), path)

    ranked = sorted(set(candidates), key=_base_priority)
    if ranked:
        return ranked[0]

    print("[LOAD] No paired BASE model found for DFlash draft. Checked:")
    print(f"    LOCAL_BASE_DIR -> {resolve_checkpoint_dir(LOCAL_BASE_DIR) or 'MISSING'}")
    parent = os.path.dirname(dflash_path)
    if os.path.isdir(parent):
        try:
            for entry in sorted(os.listdir(parent)):
                print(f"    sibling: {os.path.join(parent, entry)}")
        except OSError:
            pass
    return None


def resolve_generation_model_path(requested_path: str) -> Tuple[str, Optional[str]]:
    """
    Return (generation_path, optional_dflash_draft_path).
    DFlash draft dirs are rerouted to their paired base causal LM.
    """
    resolved = resolve_checkpoint_dir(requested_path) or requested_path
    if not local_dir_exists(resolved):
        return requested_path, None

    role = bundle_folder_role(resolved)
    if role == "base":
        draft = find_paired_dflash_draft(resolved)
        print(f"[LOAD] Base Qwen checkpoint: {resolved}")
        if draft:
            print(f"[LOAD] Paired DFlash draft: {draft}")
        return resolved, draft

    if is_full_causal_lm_checkpoint(resolved):
        draft = find_paired_dflash_draft(resolved)
        return resolved, draft

    if not is_dflash_draft_checkpoint(resolved):
        return resolved, None

    base = find_paired_base_checkpoint(resolved)
    if base:
        print(f"[LOAD] DFlash DRAFT checkpoint: {resolved}")
        print(f"[LOAD] Generation will use paired BASE model: {base}")
        return base, resolved

    raise RuntimeError(
        "[LOAD] Found a DFlash draft checkpoint but no paired base model.\n"
        f"  Draft: {resolved}\n"
        "  DFlash drafts lack embed_tokens / lm_head and cannot generate text alone.\n"
        "  Fix: place the base Qwen checkpoint beside the draft (e.g. .../Qwen3.5-4B)\n"
        f"  or set LOCAL_BASE_DIR (currently {LOCAL_BASE_DIR!r})."
    )


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
    if PREFER_HF_INFERENCE:
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
    if any(tag in path_l for tag in ("dflash", "qwen3", "custom")):
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


def _build_vllm_attempts(
    model_path: str, gpu_memory_utilization: float
) -> List[Dict[str, Any]]:
    custom = _needs_custom_vllm_handling(model_path)
    base: Dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": "auto",
        "max_model_len": 4096,
        "gpu_memory_utilization": gpu_memory_utilization,
    }

    hf_overrides: Dict[str, Any] = {}
    if os.path.isfile(os.path.join(model_path, "config.json")):
        try:
            cfg = _read_hf_config(model_path)
            archs = [str(a) for a in (cfg.get("architectures") or [])]
            print(f"[LOAD] HF config architectures: {archs}")
            if is_dflash_draft_checkpoint(model_path):
                print("[LOAD] Skipping architecture rewrite for DFlash draft checkpoint.")
            elif any("Embedding" in a or "Pooling" in a for a in archs):
                guessed = _guess_causal_lm_architecture(cfg)
                if guessed:
                    hf_overrides["architectures"] = [guessed]
                    print(f"[LOAD] Rewriting architectures for vLLM -> {[guessed]}")
        except Exception as exc:
            print(f"[LOAD] Could not inspect config.json: {exc}")

    attempts: List[Dict[str, Any]] = []
    if custom or hf_overrides or VLLM_ENFORCE_EAGER:
        attempts.append(
            {
                **base,
                "runner": VLLM_RUNNER,
                "enforce_eager": True,
                **({"hf_overrides": hf_overrides} if hf_overrides else {}),
            }
        )
    attempts.append(
        {
            **base,
            "runner": VLLM_RUNNER,
            "enforce_eager": VLLM_ENFORCE_EAGER,
            **({"hf_overrides": hf_overrides} if hf_overrides else {}),
        }
    )
    attempts.append({**base, "enforce_eager": True})
    return attempts


@dataclass
class _HFCompletionOutput:
    token_ids: List[int]
    text: str
    logprobs: Optional[List[Dict[int, float]]] = None


@dataclass
class _HFRequestOutput:
    outputs: List[_HFCompletionOutput]
    prompt_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class HFGenerateEngine:
    """Fallback engine when vLLM cannot load a custom DFlash / Qwen3.5 checkpoint."""

    def __init__(
        self,
        model_path: str,
        tokenizer: Any,
        *,
        dflash_draft_path: Optional[str] = None,
        local_only: bool = True,
    ):
        if is_dflash_draft_checkpoint(model_path):
            raise ValueError(
                f"Refusing to load DFlash draft as causal LM: {model_path}"
            )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[LOAD][HF] Loading causal LM from {model_path} ...")
        if dflash_draft_path:
            print(f"[LOAD][HF] (DFlash draft reserved for future speculative path: {dflash_draft_path})")
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.dflash_draft_path = dflash_draft_path
        self.model = _load_hf_generation_model(
            model_path,
            dtype=dtype,
            local_only=local_only,
        ).eval()
        self.device = next(self.model.parameters()).device
        emb = self.model.get_input_embeddings()
        if emb is None or emb.weight.numel() == 0:
            raise RuntimeError(
                f"[LOAD][HF] Model at {model_path} has no usable embed_tokens — "
                "this is likely a DFlash draft checkpoint, not a base causal LM."
            )
        print("[LOAD][HF] HuggingFace generate engine ready (vLLM-compatible API).")

    def _encode(self, text: str) -> torch.Tensor:
        prompt = text if text and str(text).strip() else " "
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=4096,
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

    def _prompt_logprobs_for_ids(self, input_ids: torch.Tensor) -> List[Optional[Dict[int, float]]]:
        out: List[Optional[Dict[int, float]]] = [None]
        if input_ids.shape[1] < 2:
            return out
        with torch.inference_mode():
            logits = self.model(input_ids).logits[0]
        for pos in range(1, input_ids.shape[1]):
            token_id = int(input_ids[0, pos].item())
            logp = torch.log_softmax(logits[pos - 1], dim=-1)[token_id].item()
            out.append({token_id: logp})
        return out

    def _sample_ids(
        self, input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float
    ) -> Tuple[List[int], List[Dict[int, float]]]:
        if input_ids.shape[1] == 0:
            raise ValueError("Refusing to generate from empty input_ids")

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

        for prompt, sp in zip(prompts, sp_list):
            max_tokens = int(getattr(sp, "max_tokens", 0) or 0)
            temperature = float(getattr(sp, "temperature", 1.0) or 0.0)
            top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
            want_prompt_logprobs = bool(getattr(sp, "prompt_logprobs", False))

            input_ids = self._encode(prompt)
            prompt_logprobs = (
                self._prompt_logprobs_for_ids(input_ids) if want_prompt_logprobs else None
            )
            if max_tokens <= 0:
                results.append(_HFRequestOutput(outputs=[], prompt_logprobs=prompt_logprobs))
                continue
            new_ids, step_logprobs = self._sample_ids(
                input_ids, max_tokens=max_tokens, temperature=temperature, top_p=top_p
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
    gpu_memory_utilization: float = 0.88,
    dflash_draft_path: Optional[str] = None,
) -> Any:
    """Create vLLM engine with custom-checkpoint safeguards; HF fallback on failure."""
    gen_path, auto_draft = resolve_generation_model_path(model_path)
    draft_path = dflash_draft_path or auto_draft
    local_only = os.path.isdir(gen_path)

    if _should_skip_vllm(gen_path):
        reason = (
            "PREFER_HF_INFERENCE=True"
            if PREFER_HF_INFERENCE
            else "Qwen3.5 multimodal checkpoint"
        )
        print(f"[LOAD] Skipping vLLM ({reason}); using HuggingFace generate engine.")
        return HFGenerateEngine(
            gen_path,
            tokenizer,
            dflash_draft_path=draft_path,
            local_only=local_only,
        )

    attempts = _build_vllm_attempts(gen_path, gpu_memory_utilization)
    last_err: Optional[BaseException] = None
    for idx, kwargs in enumerate(attempts, start=1):
        try:
            if idx > 1:
                print(f"[LOAD] vLLM retry {idx}/{len(attempts)}: {kwargs}")
            llm = LLM(**kwargs)
            print("[LOAD] vLLM engine ready.")
            if draft_path:
                setattr(llm, "dflash_draft_path", draft_path)
            return llm
        except Exception as exc:
            last_err = exc
            print(f"[LOAD] vLLM attempt {idx} failed: {type(exc).__name__}: {exc}")

    if VLLM_FALLBACK_TO_HF:
        print(
            "[LOAD] All vLLM attempts failed; using HuggingFace generate fallback."
        )
        return HFGenerateEngine(
            gen_path,
            tokenizer,
            dflash_draft_path=draft_path,
            local_only=local_only,
        )

    raise RuntimeError(
        f"Failed to load inference engine for {gen_path}"
    ) from last_err


def verify_inference_engine(llm: Any, tokenizer: Any) -> None:
    """
    Fail fast before ARC eval if the engine cannot tokenize/generate.
    Catches stale Kaggle copies (old HFGenerateEngine manual-forward loop) and
    DFlash drafts mistakenly loaded as causal LMs.
    """
    engine_label = (
        "HF-fallback"
        if isinstance(llm, HFGenerateEngine)
        else getattr(type(llm), "__name__", str(type(llm)))
    )
    gen_path = getattr(llm, "model_path", None)
    draft_path = getattr(llm, "dflash_draft_path", None)
    print("\n[VERIFY] Inference engine smoke test")
    print(f"    script   : {SCRIPT_VERSION}")
    print(f"    engine   : {engine_label}")
    if gen_path:
        print(f"    generate : {gen_path}")
    if draft_path:
        print(f"    dflash   : {draft_path} (draft only — not used for generation)")

    probe = "ARC smoke test."
    encoded = tokenizer(probe, return_tensors="pt", add_special_tokens=True)
    if encoded["input_ids"].shape[1] == 0:
        raise RuntimeError(
            "[VERIFY] Tokenizer produced empty input_ids. "
            "Load tokenizer from the paired BASE model, not the DFlash draft."
        )

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    try:
        out = llm.generate([probe], sp)[0]
    except Exception as exc:
        hint = ""
        if "cannot reshape tensor of 0 elements" in str(exc):
            hint = (
                "\n  Likely cause: DFlash draft loaded as causal LM, or stale script "
                f"(need SCRIPT_VERSION={SCRIPT_VERSION} in banner)."
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


def load_models(model_name: str):
    """
    Load one inference engine (vLLM preferred; HF fallback for custom DFlash checkpoints).
    Also load a lightweight HF tokenizer copy for decode/encode consistency.
    """
    local_only = os.path.isdir(model_name) and local_dir_exists(model_name)
    if local_only:
        gen_path, draft_path = resolve_generation_model_path(model_name)
    else:
        gen_path, draft_path = model_name, None
    load_label = gen_path if local_only else f"{gen_path} (remote)"
    print(
        f"\n[LOAD] Initializing inference engine for {load_label}"
        + ("" if local_only else " (this can take 30-120s on first download)...")
    )
    if draft_path:
        print(f"[LOAD] DFlash draft reserved: {draft_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        gen_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = create_inference_engine(
        gen_path,
        tokenizer,
        gpu_memory_utilization=0.88,
        dflash_draft_path=draft_path,
    )
    if isinstance(llm, HFGenerateEngine):
        setattr(llm, "model_path", gen_path)
    elif draft_path:
        setattr(llm, "dflash_draft_path", draft_path)
    setattr(llm, "generation_model_path", gen_path)

    hf_model = None
    if isinstance(llm, HFGenerateEngine):
        hf_model = llm.model
    else:
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 14 * (1024**3):
                hf_model = _load_hf_generation_model(
                    gen_path,
                    dtype=torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16,
                    local_only=local_only,
                ).eval()
                print(
                    "[LOAD] Optional HF reference model also resident for hidden-state introspection."
                )
        except Exception:
            pass

    return llm, tokenizer, hf_model


# =============================================================================
# LOCAL KAGGLE LOADERS (optional dual-engine path when both checkpoints exist)
# =============================================================================
def load_local_models() -> Tuple[LLM, Any, Optional[LLM], Optional[Any]]:
    """
    Load local checkpoints without network access.

    DFlash draft checkpoints are NEVER used for generation directly.
    A paired BASE causal LM must exist (LOCAL_BASE_DIR or sibling folder).

    Returns: (primary_llm, primary_tokenizer, optional_second_llm, optional_second_tokenizer)
    """
    print("\n[LOCAL-LOAD] Checkpoint inventory:")
    print(f"    ROOT   {KAGGLE_QWEN_BUNDLE_ROOT} -> "
          f"{'OK' if os.path.isdir(KAGGLE_QWEN_BUNDLE_ROOT) else 'MISSING'}")
    for tag, raw in (
        ("BASE", LOCAL_BASE_DIR),
        ("DFLASH", LOCAL_DFLASH_DIR),
        ("MODEL", LOCAL_MODEL_PATH),
    ):
        resolved = resolve_checkpoint_dir(raw) if raw else None
        role = "n/a"
        if resolved:
            named = bundle_folder_role(resolved)
            role = (
                "BASE" if named == "base"
                else "DRAFT" if named == "draft"
                else (
                    "FULL"
                    if is_full_causal_lm_checkpoint(resolved)
                    else ("DRAFT" if is_dflash_draft_checkpoint(resolved) else "UNKNOWN")
                )
            )
        print(f"    {tag:<6} {raw or '—'} -> {resolved or 'MISSING'} [{role}]")
        if resolved:
            print_checkpoint_diagnostics(resolved)

    bundle_base, bundle_dflash = discover_qwen_dflash_bundle_paths()
    if bundle_base or bundle_dflash:
        print("[LOCAL-LOAD] Auto-discovered qwen3.5-4b-dflash bundle:")
        print(f"    BASE   -> {bundle_base or 'MISSING'}")
        print(f"    DFLASH -> {bundle_dflash or 'MISSING'}")

    def _load_generation_engine(
        gen_path: str,
        *,
        label: str,
        gpu_util: float,
        draft_path: Optional[str] = None,
    ) -> Tuple[LLM, Any]:
        print(f"\n[LOCAL-LOAD] {label}")
        print(f"    generation: {gen_path}")
        if draft_path:
            print(f"    dflash    : {draft_path} (draft — not used for token generation)")
        tokenizer = AutoTokenizer.from_pretrained(
            gen_path, trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = create_inference_engine(
            gen_path,
            tokenizer,
            gpu_memory_utilization=gpu_util,
            dflash_draft_path=draft_path,
        )
        if isinstance(llm, HFGenerateEngine):
            setattr(llm, "model_path", gen_path)
        setattr(llm, "generation_model_path", gen_path)
        backend = "HF-fallback" if isinstance(llm, HFGenerateEngine) else "vLLM"
        print(
            f"[LOCAL-LOAD] {os.path.basename(gen_path)} engine ready ({backend})."
        )
        return llm, tokenizer

    # Always load Qwen3.5-4B as the base/generation model — never classify it as draft.
    base_path = resolve_checkpoint_dir(LOCAL_BASE_DIR)
    if not base_path and os.path.isdir(KAGGLE_QWEN_BUNDLE_ROOT):
        base_path = os.path.join(KAGGLE_QWEN_BUNDLE_ROOT, "Qwen3.5-4B")
        if not local_dir_exists(base_path):
            base_path = None

    if base_path and local_dir_exists(base_path):
        dflash_path = resolve_checkpoint_dir(LOCAL_DFLASH_DIR)
        if not dflash_path and os.path.isdir(KAGGLE_QWEN_BUNDLE_ROOT):
            candidate = os.path.join(KAGGLE_QWEN_BUNDLE_ROOT, "Qwen3.5-4B-DFlash")
            if local_dir_exists(candidate):
                dflash_path = candidate
        gen_llm, gen_tok = _load_generation_engine(
            base_path,
            label="Base Qwen3.5-4B (generation)",
            gpu_util=0.85,
            draft_path=dflash_path if dflash_path and is_dflash_draft_checkpoint(dflash_path) else None,
        )
        if dflash_path:
            setattr(gen_llm, "dflash_draft_path", dflash_path)
        return gen_llm, gen_tok, None, None

    ensure_model_available()
    primary = resolve_local_model_path()
    if primary is None:
        raise RuntimeError(
            "[LOCAL-LOAD] No usable local checkpoint found after prefetch.\n"
            "  Expected one of:\n"
            f"    - {LOCAL_DFLASH_DIR}\n"
            f"    - {LOCAL_BASE_DIR}\n"
            f"    - LOCAL_MODEL_PATH = {LOCAL_MODEL_PATH!r}\n"
            "    - any */config.json under /kaggle/input, /kaggle/working, or HF hub cache\n"
            "  Turn Kaggle Internet ON and re-run from the top, or add dataset "
            f"'{KAGGLE_DATASET_HANDLE}' as notebook Input.\n"
            "  If using DFlash: also attach the paired BASE model at LOCAL_BASE_DIR "
            f"({LOCAL_BASE_DIR!r})."
        )

    gen_path, draft_path = resolve_generation_model_path(primary)
    llm, tok = _load_generation_engine(
        gen_path,
        label="Primary local checkpoint",
        gpu_util=0.85,
        draft_path=draft_path,
    )
    return llm, tok, None, None


# =============================================================================
# ARC-AGI DATASET EVALUATION (parameterized paths; data/ is not in git)
# =============================================================================
ARC_SPLIT_FILES = {
    "training": (
        "arc-agi_training_challenges.json",
        "arc-agi_training_solutions.json",
    ),
    "evaluation": (
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json",
    ),
}


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


def format_arc_prompt(task_id: str, task: Dict[str, Any], test_index: int = 0) -> str:
    """Turn one ARC task into a text prompt with train demos + one test input."""
    lines = [
        f"ARC-AGI task {task_id}. Infer the grid transformation from the training pairs.",
        "Output format: a single JSON 2D array of integers (cell colors 0-9 only).",
        "Example: [[0,1,2],[3,4,5]]",
        "Do not include markdown fences, explanations, or any text before/after the array.",
    ]
    for i, example in enumerate(task.get("train", []), start=1):
        lines.append(f"Train {i} input: {json.dumps(example['input'])}")
        lines.append(f"Train {i} output: {json.dumps(example['output'])}")
    test_inputs = task.get("test", [])
    if test_index >= len(test_inputs):
        raise IndexError(
            f"Task {task_id} has {len(test_inputs)} test inputs; requested index {test_index}."
        )
    lines.append(f"Test input: {json.dumps(test_inputs[test_index]['input'])}")
    lines.append("Test output JSON array:")
    return "\n".join(lines)


def apply_arc_chat_prompt(tokenizer: Any, user_content: str) -> str:
    """Wrap ARC content with the model's chat template (required for Qwen3.5)."""
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You solve ARC-AGI grid puzzles. Reply with exactly one JSON 2D array "
                "of integers (values 0-9). No markdown, no explanation, no thinking."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    template_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if ARC_DISABLE_THINKING:
        template_kwargs["enable_thinking"] = False
    if ARC_ASSISTANT_PREFILL:
        messages.append({"role": "assistant", "content": ARC_ASSISTANT_PREFILL})
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
    if not isinstance(parsed, list) or not parsed:
        return False
    if not all(isinstance(row, list) for row in parsed):
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
    gen = _strip_model_artifacts(gen)
    if ARC_ASSISTANT_PREFILL and gen and not gen.lstrip().startswith("["):
        gen = ARC_ASSISTANT_PREFILL + gen
    return gen


def arc_direct_generate(
    vllm_llm: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    *,
    temperature: float = ARC_GENERATION_TEMPERATURE,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Single-pass generation for ARC eval (recommended on HF fallback).
    Avoids the classic speculative loop that rejects tokens when draft≠target sampling.
    """
    del seed  # temperature 0 is deterministic; seed reserved for API compatibility
    sp = SamplingParams(
        temperature=float(temperature),
        top_p=1.0,
        max_tokens=int(max_new_tokens),
    )
    out = vllm_llm.generate([prompt], sp)[0]
    gen_ids: List[int] = []
    if out.outputs:
        gen_ids = list(out.outputs[0].token_ids)
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    result: Dict[str, Any] = {
        "final_text": prompt + generated_text,
        "generated_text": generated_text,
        "generated_ids": gen_ids,
        "num_tokens": len(gen_ids),
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
    parsed_grid = parse_grid_from_text(gen_suffix, only_after=None)
    if parsed_grid is not None:
        result["parsed_grid"] = parsed_grid
        result["generated_text"] = json.dumps(parsed_grid, separators=(",", ":"))
    return result


def arc_classic_generate(
    vllm_llm: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    block_size: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Dispatch classic ARC generation: direct (default) or speculative dflash loop."""
    if ARC_EVAL_GENERATION_MODE == "direct":
        return arc_direct_generate(
            vllm_llm, tokenizer, prompt, max_new_tokens=max_new_tokens, seed=seed
        )
    res = classic_dflash_generate(
        vllm_llm,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        seed=seed,
    )
    res["generation_mode"] = "speculative"
    if "generated_text" not in res:
        res["generated_text"] = extract_arc_generated_suffix(res, prompt)
    return res


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


def generation_timing_stats(elapsed_s: float, num_tokens: int) -> Dict[str, Any]:
    """Elapsed wall time + tokens/sec for one generation call."""
    elapsed = max(0.0, float(elapsed_s))
    tokens = max(0, int(num_tokens or 0))
    return {
        "elapsed_s": elapsed,
        "num_tokens": tokens,
        "tps": tokens / max(elapsed, 1e-6),
    }


def format_timing_line(timing: Dict[str, Any]) -> str:
    return (
        f"{timing['num_tokens']} tokens in {timing['elapsed_s']:.2f}s "
        f"({timing['tps']:.2f} tok/s)"
    )


def arc_eval_log(message: str) -> None:
    """Print + flush so Kaggle notebooks show ARC progress immediately."""
    print(message, flush=True)


def print_arc_answer_comparison(
    *,
    task_id: str,
    task_idx: int,
    num_tasks: int,
    test_index: int,
    gold: List[List[int]],
    classic_pred: Optional[List[List[int]]],
    mbr_pred: Optional[List[List[int]]],
    classic_stats: Dict[str, Any],
    mbr_stats: Dict[str, Any],
    classic_raw_text: Optional[str] = None,
    mbr_raw_text: Optional[str] = None,
    classic_timing: Optional[Dict[str, Any]] = None,
    mbr_timing: Optional[Dict[str, Any]] = None,
    split: str = ARC_DATA_SPLIT,
) -> None:
    """Print ground truth and both model answers side-by-side (JSON grids)."""
    bar = "=" * 80
    print(f"\n{bar}")
    print(
        f"ARC ANSWER vs GROUND TRUTH  |  task {task_idx + 1}/{num_tasks}  "
        f"|  id={task_id}  |  split={split}  |  test #{test_index}"
    )
    print(bar)

    print("\nGROUND TRUTH (gold labels):")
    print(format_grid_json(gold))

    c_tag = _grade_verdict_label(classic_stats)
    c_time = (
        f"  |  {format_timing_line(classic_timing)}"
        if classic_timing
        else ""
    )
    print(f"\nCLASSIC PREDICTION [{c_tag}] "
          f"({classic_stats['matching_cells']}/{classic_stats['gold_cells']} cells, "
          f"{classic_stats['match_rate'] * 100:.1f}% match){c_time}:")
    print(format_grid_json(classic_pred))
    if classic_pred is None and classic_raw_text:
        tail = classic_raw_text[-800:] if len(classic_raw_text) > 800 else classic_raw_text
        print("CLASSIC raw model output (tail):")
        print(tail)

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

    if classic_pred and gold:
        print("\nCELL-BY-CELL (classic vs gold) — XX=mismatch, ok=match:")
        for r in range(len(gold)):
            row_markers = []
            for c in range(len(gold[r])):
                if r >= len(classic_pred) or c >= len(classic_pred[r]):
                    row_markers.append("??")
                elif classic_pred[r][c] == gold[r][c]:
                    row_markers.append("ok")
                else:
                    row_markers.append("XX")
            print(f"  row {r}: " + " ".join(row_markers))

    if mbr_pred and gold:
        print("\nCELL-BY-CELL (million-brains vs gold) — XX=mismatch, ok=match:")
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


def print_arc_classic_interim(
    *,
    task_id: str,
    task_idx: int,
    num_tasks: int,
    test_index: int,
    gold: List[List[int]],
    classic_pred: Optional[List[List[int]]],
    classic_stats: Dict[str, Any],
    classic_timing: Dict[str, Any],
    classic_raw_tail: Optional[str] = None,
    split: str = ARC_DATA_SPLIT,
) -> None:
    """Print classic result + timing immediately (before slow MBR pass finishes)."""
    c_tag = _grade_verdict_label(classic_stats)
    arc_eval_log(
        f"\n[ARC] {task_idx + 1}/{num_tasks} {task_id} test#{test_index} "
        f"— CLASSIC done [{c_tag}] | {format_timing_line(classic_timing)}"
    )
    arc_eval_log(f"  GOLD    : {format_grid_json(gold)}")
    arc_eval_log(
        f"  CLASSIC : {format_grid_json(classic_pred)} "
        f"({classic_stats['matching_cells']}/{classic_stats['gold_cells']} cells)"
    )
    if classic_raw_tail and not classic_stats.get("parsed"):
        last_anchor = classic_raw_tail.rfind("[[")
        snippet = (
            classic_raw_tail[last_anchor : last_anchor + 800]
            if last_anchor >= 0
            else classic_raw_tail[-800:]
        )
        arc_eval_log(f"  CLASSIC raw (unparsed, last [[ attempt): {snippet}")
    if ARC_RUN_MBR_EVAL:
        arc_eval_log("  (running million-brains pass next…)")


def print_arc_full_answer_report(
    comparisons: List[Dict[str, Any]], split: str
) -> None:
    """Final digest: every task's predictions vs ground truth."""
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"ARC FULL ANSWER REPORT  ({split} split — all tasks vs ground truth)")
    print(bar)
    print(
        f"{'Task':<14} {'Test':>4}  {'Classic':<10} {'MBR':<10}  "
        f"{'Classic TPS':>12}  {'MBR TPS':>10}  Gold"
    )
    print("-" * 80)
    for rec in comparisons:
        gold_s = rec.get("gold_json", "[]")
        if len(gold_s) > 36:
            gold_s = gold_s[:33] + "..."
        c_timing = rec.get("classic_timing") or {}
        m_timing = rec.get("mbr_timing") or {}
        c_tps = f"{c_timing.get('tps', 0.0):.2f} tok/s"
        m_tps = f"{m_timing.get('tps', 0.0):.2f} tok/s"
        c_elapsed = f"{c_timing.get('elapsed_s', 0.0):.1f}s"
        m_elapsed = f"{m_timing.get('elapsed_s', 0.0):.1f}s"
        print(
            f"{rec['task_id']:<14} {rec['test_index']:>4}  "
            f"{rec['classic_verdict']:<10} {rec['mbr_verdict']:<10}  "
            f"{c_tps:>6} ({c_elapsed:>5})  {m_tps:>6} ({m_elapsed:>5})  {gold_s}"
        )
        classic_s = rec.get("classic_json", "(unparsed)")
        mbr_s = rec.get("mbr_json", "(unparsed)")
        if len(classic_s) > 76:
            classic_s = classic_s[:73] + "..."
        if len(mbr_s) > 76:
            mbr_s = mbr_s[:73] + "..."
        print(f"    gold   : {rec.get('gold_json', '[]')}")
        print(
            f"    classic: {classic_s}  "
            f"[{format_timing_line(c_timing) if c_timing else 'n/a'}]"
        )
        print(
            f"    mbr    : {mbr_s}  "
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
    classic_pred: Optional[List[List[int]]],
    mbr_pred: Optional[List[List[int]]],
    classic_stats: Dict[str, Any],
    mbr_stats: Dict[str, Any],
    split: str = ARC_DATA_SPLIT,
) -> None:
    """Terminal grade card: test input vs gold answer key vs both predictions."""
    c_verdict = _grade_verdict_label(classic_stats)
    m_verdict = _grade_verdict_label(mbr_stats)
    c_icon = "✓" if classic_stats["correct"] else "✗"
    m_icon = "✓" if mbr_stats["correct"] else "✗"

    bar = "═" * 78
    print(f"\n╔{bar}╗")
    print(
        f"║  ARC GRADE  task {task_idx + 1}/{num_tasks}  id={task_id}  "
        f"split={split}  test=#{test_index}"
        f"{' ' * max(0, 18 - len(task_id))}║"
    )
    print(
        f"║  CLASSIC {c_icon} {c_verdict:<14}  |  "
        f"MILLION-BRAINS {m_icon} {m_verdict:<14}"
        f"{' ' * 22}║"
    )
    print(f"╠{bar}╣")

    blocks = [
        _render_grid_ascii(test_input, title="TEST INPUT (challenge)"),
        _render_grid_ascii(gold, title="GOLD SOLUTION (test set labels)"),
        _render_grid_ascii(
            classic_pred if classic_pred else [[-1]],
            title=f"CLASSIC PRED [{c_verdict}]",
        )
        if classic_pred
        else ["CLASSIC PRED [UNPARSED]", "  (model output not a valid grid)"],
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
        f"║  CLASSIC match: {classic_stats['matching_cells']:>3}/{classic_stats['gold_cells']:<3} "
        f"({classic_stats['match_rate'] * 100:5.1f}%)  "
        f"shape {classic_stats['pred_shape'] or '—'} vs gold {classic_stats['gold_shape']}"
        f"{' ' * 8}║"
    )
    print(
        f"║  MBR match:     {mbr_stats['matching_cells']:>3}/{mbr_stats['gold_cells']:<3} "
        f"({mbr_stats['match_rate'] * 100:5.1f}%)  "
        f"shape {mbr_stats['pred_shape'] or '—'} vs gold {mbr_stats['gold_shape']}"
        f"{' ' * 8}║"
    )
    print(f"╚{bar}╝")

    if classic_pred and classic_stats["shape_match"]:
        diff_lines = _align_grid_columns(
            [
                _render_grid_ascii(
                    gold,
                    title="DIFF vs GOLD (classic)",
                    diff_against=gold,
                    pred_for_diff=classic_pred,
                ),
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
    classic_pred: Optional[List[List[int]]],
    mbr_pred: Optional[List[List[int]]],
    classic_stats: Dict[str, Any],
    mbr_stats: Dict[str, Any],
    train_pairs: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Save a PNG grade card contrasting predictions against the test-set gold grid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    c_verdict = _grade_verdict_label(classic_stats)
    m_verdict = _grade_verdict_label(mbr_stats)
    n_train = min(len(train_pairs or []), 3) if ARC_SHOW_TRAIN_EXAMPLES else 0
    ncols = 5 + n_train
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
    if classic_pred:
        _draw_grid_matplotlib(axes[col], classic_pred, f"CLASSIC [{c_verdict}]")
    else:
        axes[col].text(0.5, 0.5, "UNPARSED", ha="center", va="center", fontsize=12)
        axes[col].set_title(f"CLASSIC [{c_verdict}]")
        axes[col].axis("off")
    col += 1
    if mbr_pred:
        _draw_grid_matplotlib(axes[col], mbr_pred, f"MBR [{m_verdict}]")
    else:
        axes[col].text(0.5, 0.5, "UNPARSED", ha="center", va="center", fontsize=12)
        axes[col].set_title(f"MBR [{m_verdict}]")
        axes[col].axis("off")
    col += 1
    _draw_diff_matplotlib(
        axes[col],
        classic_pred,
        gold,
        f"Classic diff ({classic_stats['match_rate'] * 100:.0f}%)",
    )

    c_color = "#2ecc40" if classic_stats["correct"] else "#ff4136"
    m_color = "#2ecc40" if mbr_stats["correct"] else "#ff4136"
    fig.suptitle(
        f"ARC {task_id} test #{test_index}  |  "
        f"Classic: {c_verdict}  |  MBR: {m_verdict}",
        fontsize=11,
        fontweight="bold",
        color="#222222",
    )
    fig.text(
        0.5,
        0.02,
        f"Classic {classic_stats['matching_cells']}/{classic_stats['gold_cells']} cells  "
        f"|  MBR {mbr_stats['matching_cells']}/{mbr_stats['gold_cells']} cells",
        ha="center",
        fontsize=9,
    )
    classic_ax_idx = n_train + 2
    mbr_ax_idx = n_train + 3
    for ax, color in (
        (axes[classic_ax_idx], c_color),
        (axes[mbr_ax_idx], m_color),
    ):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
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
    classic_pred: Optional[List[List[int]]],
    mbr_pred: Optional[List[List[int]]],
    train_pairs: Optional[List[Dict[str, Any]]] = None,
    split: str = ARC_DATA_SPLIT,
) -> Dict[str, Any]:
    """Full visual grading for one test case: stats + terminal card + optional PNG."""
    classic_stats = grid_cell_stats(classic_pred, gold)
    mbr_stats = grid_cell_stats(mbr_pred, gold)

    if ARC_VISUAL_GRADING:
        print_arc_grade_card(
            task_id=task_id,
            task_idx=task_idx,
            num_tasks=num_tasks,
            test_index=test_index,
            test_input=test_input,
            gold=gold,
            classic_pred=classic_pred,
            mbr_pred=mbr_pred,
            classic_stats=classic_stats,
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
            classic_pred=classic_pred,
            mbr_pred=mbr_pred,
            classic_stats=classic_stats,
            mbr_stats=mbr_stats,
            train_pairs=train_pairs,
        )
        if image_path:
            display_arc_grade_image(image_path)

    return {
        "classic_stats": classic_stats,
        "mbr_stats": mbr_stats,
        "image_path": image_path,
    }


def print_arc_gradeboard(per_task: List[Dict[str, Any]], split: str) -> None:
    """Final at-a-glance scoreboard for every challenged task."""
    print("\n" + "━" * 78)
    print(f"ARC GRADEBOARD  ({split} split — contrasted against test-set gold labels)")
    print("━" * 78)
    print(f"{'Task ID':<12} {'Tests':>5}  {'Classic':>16}  {'Million-Brains':>16}")
    print("-" * 78)
    for rec in per_task:
        tid = rec["task_id"]
        tests = len(rec.get("classic", []))
        c_pass = sum(1 for r in rec["classic"] if r.get("correct"))
        m_pass = sum(1 for r in rec["mbr"] if r.get("correct"))
        c_bar = "█" * c_pass + "░" * max(0, tests - c_pass)
        m_bar = "█" * m_pass + "░" * max(0, tests - m_pass)
        print(
            f"{tid:<12} {tests:>5}  {c_pass}/{tests} {c_bar:<8}  {m_pass}/{tests} {m_bar}"
        )
    print("━" * 78)


def evaluate_arc_dataset(
    vllm_llm: LLM,
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
    Run classic vs million-brains-dflash on an ARC-AGI split.
    Requires explicit challenges + solutions paths (typically under data/, gitignored).
    """
    dataset = load_arc_dataset(challenges_path, solutions_path)
    challenges = dataset["challenges"]
    solutions = dataset["solutions"]

    task_ids = sorted(challenges.keys())
    if max_tasks is not None:
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
    print(
        f"ARC generation mode: {ARC_EVAL_GENERATION_MODE} "
        f"(chat_template={ARC_USE_CHAT_TEMPLATE}, temp={ARC_GENERATION_TEMPERATURE})"
    )
    print(f"ARC run MBR eval: {ARC_RUN_MBR_EVAL}")
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
        "classic": {"correct": 0, "parsed": 0, "tests": 0, "time": 0.0, "tokens": 0},
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
            "classic": [],
            "mbr": [],
        }

        for test_index in range(num_tests):
            prompt = build_arc_inference_prompt(
                tokenizer, task_id, task, test_index=test_index
            )
            gold = gold_tests[test_index]
            test_input = task["test"][test_index]["input"]

            arc_eval_log(
                f"\n[ARC] >>> task {task_idx + 1}/{len(task_ids)} {task_id} "
                f"test#{test_index} — starting CLASSIC pass "
                f"({ARC_EVAL_GENERATION_MODE})…"
            )
            t0 = time.perf_counter()
            classic_res = arc_classic_generate(
                vllm_llm,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                block_size=max(6, block_size),
                seed=seed + test_index,
            )
            classic_elapsed = time.perf_counter() - t0
            classic_timing = generation_timing_stats(
                classic_elapsed, classic_res["num_tokens"]
            )
            summary["classic"]["time"] += classic_elapsed
            summary["classic"]["tokens"] += classic_res["num_tokens"]
            summary["classic"]["tests"] += 1

            classic_answer_text = extract_arc_generated_suffix(classic_res, prompt)
            classic_pred = parse_grid_from_text(classic_answer_text, only_after=None)
            classic_stats_early = grid_cell_stats(classic_pred, gold)

            if ARC_PRINT_ALL_ANSWERS:
                print_arc_classic_interim(
                    task_id=task_id,
                    task_idx=task_idx,
                    num_tasks=len(task_ids),
                    test_index=test_index,
                    gold=gold,
                    classic_pred=classic_pred,
                    classic_stats=classic_stats_early,
                    classic_timing=classic_timing,
                    classic_raw_tail=classic_answer_text,
                    split=split,
                )

            if ARC_RUN_MBR_EVAL:
                arc_eval_log(
                    f"[ARC] >>> task {task_idx + 1}/{len(task_ids)} {task_id} "
                    f"test#{test_index} — starting MILLION-BRAINS pass…"
                )
                t0 = time.perf_counter()
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
                    verbose=ARC_EVAL_VERBOSE,
                )
                mbr_elapsed = time.perf_counter() - t0
                mbr_timing = generation_timing_stats(mbr_elapsed, mbr_res["num_tokens"])
                summary["mbr"]["time"] += mbr_elapsed
                summary["mbr"]["tokens"] += mbr_res["num_tokens"]
                mbr_answer_text = extract_arc_generated_suffix(mbr_res, prompt)
                mbr_pred = parse_grid_from_text(mbr_answer_text, only_after=None)
                arc_eval_log(
                    f"[ARC] <<< task {task_idx + 1}/{len(task_ids)} {task_id} "
                    f"test#{test_index} — MBR done | {format_timing_line(mbr_timing)}"
                )
            else:
                mbr_res = {
                    "final_text": prompt,
                    "generated_text": "",
                    "num_tokens": 0,
                    "generation_mode": "skipped",
                }
                mbr_timing = generation_timing_stats(0.0, 0)
                mbr_answer_text = ""
                mbr_pred = None
                arc_eval_log(
                    f"[ARC] --- task {task_idx + 1}/{len(task_ids)} {task_id} "
                    f"test#{test_index} — MBR skipped (ARC_RUN_MBR_EVAL=False)"
                )
            summary["mbr"]["tests"] += 1

            grade_info = grade_arc_test_case(
                task_id=task_id,
                task_idx=task_idx,
                num_tasks=len(task_ids),
                test_index=test_index,
                test_input=test_input,
                gold=gold,
                classic_pred=classic_pred,
                mbr_pred=mbr_pred,
                train_pairs=task.get("train") if ARC_SHOW_TRAIN_EXAMPLES else None,
                split=split,
            ) if visual_grading else {
                "classic_stats": grid_cell_stats(classic_pred, gold),
                "mbr_stats": grid_cell_stats(mbr_pred, gold),
                "image_path": None,
            }

            classic_stats = grade_info["classic_stats"]
            mbr_stats = grade_info["mbr_stats"]

            comparison_rec = {
                "task_id": task_id,
                "task_idx": task_idx,
                "test_index": test_index,
                "split": split,
                "gold": gold,
                "gold_json": format_grid_json(gold),
                "classic_pred": classic_pred,
                "classic_json": format_grid_json(classic_pred),
                "mbr_pred": mbr_pred,
                "mbr_json": format_grid_json(mbr_pred),
                "classic_verdict": _grade_verdict_label(classic_stats),
                "mbr_verdict": _grade_verdict_label(mbr_stats),
                "classic_correct": classic_stats["correct"],
                "mbr_correct": mbr_stats["correct"],
                "classic_match_rate": classic_stats["match_rate"],
                "mbr_match_rate": mbr_stats["match_rate"],
                "classic_timing": classic_timing,
                "mbr_timing": mbr_timing,
                "classic_mode": classic_res.get("generation_mode"),
                "mbr_mode": mbr_res.get("generation_mode"),
            }
            summary["answer_comparisons"].append(comparison_rec)

            if ARC_PRINT_ALL_ANSWERS:
                print_arc_answer_comparison(
                    task_id=task_id,
                    task_idx=task_idx,
                    num_tasks=len(task_ids),
                    test_index=test_index,
                    gold=gold,
                    classic_pred=classic_pred,
                    mbr_pred=mbr_pred,
                    classic_stats=classic_stats,
                    mbr_stats=mbr_stats,
                    classic_raw_text=classic_answer_text,
                    mbr_raw_text=mbr_answer_text,
                    classic_timing=classic_timing,
                    mbr_timing=mbr_timing,
                    split=split,
                )

            if classic_stats["parsed"]:
                summary["classic"]["parsed"] += 1
            if classic_stats["correct"]:
                summary["classic"]["correct"] += 1
            if mbr_stats["parsed"]:
                summary["mbr"]["parsed"] += 1
            if mbr_stats["correct"]:
                summary["mbr"]["correct"] += 1

            task_record["classic"].append(
                {
                    "test_index": test_index,
                    "parsed": classic_stats["parsed"],
                    "correct": classic_stats["correct"],
                    "match_rate": classic_stats["match_rate"],
                    "matching_cells": classic_stats["matching_cells"],
                    "gold_cells": classic_stats["gold_cells"],
                    "prediction": classic_pred,
                    "gold": gold,
                    "elapsed_s": classic_timing["elapsed_s"],
                    "num_tokens": classic_timing["num_tokens"],
                    "tps": classic_timing["tps"],
                }
            )
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
        c_ok = sum(1 for r in task_record["classic"] if r["correct"])
        m_ok = sum(1 for r in task_record["mbr"] if r["correct"])
        c_task_tps = (
            sum(r.get("num_tokens", 0) for r in task_record["classic"])
            / max(1e-6, sum(r.get("elapsed_s", 0.0) for r in task_record["classic"]))
        )
        m_task_tps = (
            sum(r.get("num_tokens", 0) for r in task_record["mbr"])
            / max(1e-6, sum(r.get("elapsed_s", 0.0) for r in task_record["mbr"]))
        )
        print(
            f"[ARC] {task_idx + 1:4d}/{len(task_ids)} {task_id} "
            f"classic {c_ok}/{num_tests} ({c_task_tps:.2f} tok/s) | "
            f"mbr {m_ok}/{num_tests} ({m_task_tps:.2f} tok/s)"
        )

    def _rates(bucket: Dict[str, Any]) -> Dict[str, float]:
        tests = max(1, bucket["tests"])
        return {
            "accuracy": bucket["correct"] / tests,
            "parse_rate": bucket["parsed"] / tests,
            "tps": bucket["tokens"] / max(1e-6, bucket["time"]),
        }

    summary["classic"].update(_rates(summary["classic"]))
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
    print("ARC-AGI RESULTS")
    print("=" * 80)
    print(
        f"{'Metric':<28} {'Classic':>14} {'Million-Brains':>18}"
    )
    print("-" * 62)
    print(
        f"{'Test cases':<28} {summary['classic']['tests']:>14} {summary['mbr']['tests']:>18}"
    )
    print(
        f"{'Parsed outputs':<28} {summary['classic']['parsed']:>14} {summary['mbr']['parsed']:>18}"
    )
    print(
        f"{'Exact matches':<28} {summary['classic']['correct']:>14} {summary['mbr']['correct']:>18}"
    )
    print(
        f"{'Accuracy':<28} {summary['classic']['accuracy']:>13.2%} {summary['mbr']['accuracy']:>17.2%}"
    )
    print(
        f"{'Parse rate':<28} {summary['classic']['parse_rate']:>13.2%} {summary['mbr']['parse_rate']:>17.2%}"
    )
    print(
        f"{'Total elapsed (s)':<28} {summary['classic']['time']:>14.2f} {summary['mbr']['time']:>18.2f}"
    )
    print(
        f"{'Avg elapsed / test (s)':<28} "
        f"{summary['classic']['time'] / max(1, summary['classic']['tests']):>14.2f} "
        f"{summary['mbr']['time'] / max(1, summary['mbr']['tests']):>18.2f}"
    )
    print(
        f"{'Tokens / sec':<28} {summary['classic']['tps']:>14.2f} {summary['mbr']['tps']:>18.2f}"
    )
    print("=" * 80 + "\n")
    return summary


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="one-million-brains-dflash benchmark and ARC-AGI evaluation"
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
    """
    Head-to-head:
      - Classic Para-DFlash (K=1, plain cumprod speculative)
      - one-million-brains-dflash (K=4, dynamic permutation-based feature-slot allocation - the Fast Million Brains approach)
    Reports tokens/sec, avg accepted tokens per block, feature reallocation count,
    and sample text from both modes.
    """
    print("\n" + "=" * 80)
    print(
        "BENCHMARK: CLASSIC PARA-DFLASH vs MILLION-BRAINS-DFLASH (K=4, Fast Million Brains)"
    )
    print("=" * 80)
    print(f"Prompt (first 180 chars): {prompt[:180]}...")
    print(
        f"Target generation length: {max_new} tokens | Block size: {BLOCK_SIZE} | K: {K}"
    )
    print("-" * 80)

    # --- CLASSIC ---
    print("\n[CLASSIC] Running single-path Para-DFlash baseline ...")
    t0 = time.perf_counter()
    classic_res = classic_dflash_generate(
        vllm_llm,
        tokenizer,
        prompt,
        max_new_tokens=max_new,
        block_size=max(6, BLOCK_SIZE),
        seed=SEED,
    )
    t_classic = time.perf_counter() - t0
    classic_tps = classic_res["num_tokens"] / max(1e-6, t_classic)

    # --- MILLION-BRAINS ---
    print(
        "\n[MILLION-BRAINS] Running full one-million-brains-dflash (permutation allocator + CTSB smoothing + cross-stream integration + adaptive reallocation) ..."
    )
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

    # --- REPORT ---
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"\n{'Metric':<30} {'Classic (K=1)':>18} {'one-million-brains-dflash (K=4)':>22}"
    )
    print("-" * 72)
    print(
        f"{'Generated tokens':<30} {classic_res['num_tokens']:>18} {mbr_res['num_tokens']:>22}"
    )
    print(f"{'Wall time (s)':<30} {t_classic:>18.2f} {t_mbr:>22.2f}")
    print(f"{'Tokens / sec':<30} {classic_tps:>18.2f} {mbr_tps:>22.2f}")
    print(
        f"{'Super-blocks executed':<30} {classic_res['num_superblocks']:>18} {mbr_res['num_superblocks']:>22}"
    )
    print(
        f"{'Avg accepted tokens / block':<30} {classic_res['avg_accepted_per_block']:>18.2f} {mbr_res['avg_accepted_per_block']:>22.2f}"
    )
    print(
        f"{'Feature reallocations':<30} {classic_res.get('feature_reallocations', 0):>18} {mbr_res.get('feature_reallocations', 0):>22}"
    )
    print(
        f"{'Divergence events':<30} {classic_res['reframe_events']:>18} {mbr_res['reframe_events']:>22}"
    )
    if mbr_res.get("circuit_smoothing_enabled"):
        print(
            f"{'Avg circuit blend λ':<30} {'n/a':>18} {mbr_res.get('avg_blend_lambda', 1.0):>22.3f}"
        )

    print("\n--- Sample (Classic) ---")
    print(
        classic_res["final_text"][-600:]
        if len(classic_res["final_text"]) > 600
        else classic_res["final_text"]
    )
    print("\n--- Sample (one-million-brains-dflash) ---")
    print(
        mbr_res["final_text"][-600:]
        if len(mbr_res["final_text"]) > 600
        else mbr_res["final_text"]
    )

    # Rich diagnostic
    print("\n[MILLION-BRAINS] Feature allocation history (last 6 super-blocks):")
    for i, feats in enumerate(mbr_res["feature_history"][-6:]):
        print(f"    SB {len(mbr_res['feature_history']) - 6 + i:02d}: {feats}")

    print("\n[MILLION-BRAINS] Acceptance trajectory (per super-block):")
    print("   ", [round(a, 3) for a in mbr_res["acceptance_history"][-12:]])

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80 + "\n")

    # Return for further programmatic use if needed
    return {
        "classic": {**classic_res, "tps": classic_tps, "time": t_classic},
        "mbr": {**mbr_res, "tps": mbr_tps, "time": t_mbr},
    }


# =============================================================================
# MAIN ENTRY POINT (Kaggle script style - just run the file)
# =============================================================================
if __name__ == "__main__":
    args = parse_cli_args()

    print(
        "\n[million_brains_dflash.py] Starting full one-million-brains-dflash (Fast Million Brains) Kaggle run"
    )
    print(f"    SCRIPT_VERSION={SCRIPT_VERSION}")
    print(
        f"    K={K}, BLOCK_SIZE={BLOCK_SIZE}, ENABLE_FEATURE_REALLOCATION={ENABLE_FEATURE_REALLOCATION}"
    )
    print(f"    SEED={SEED}, TARGET_MAX_TOKENS={TARGET_MAX_TOKENS}")
    print_arc_data_config(
        args.eval_challenges, args.eval_solutions, args.arc_source
    )

    # 1) Live-edit banner was already printed right after the patcher ran.

    # 2) Optional one-time prefetch when Kaggle Internet is enabled
    ensure_model_available()

    # 3) Choose & load model (local/offline first; remote only when online)
    model_name = "unknown"
    try:
        if PREFER_LOCAL_MODELS:
            dflash_llm, dflash_tok, base_llm, base_tok = load_local_models()
            vllm_llm, tokenizer, hf_model = dflash_llm, dflash_tok, None
            model_name = resolve_local_model_path() or LOCAL_DFLASH_DIR or "local"
            if base_llm is not None:
                _available_engines = {"dflash": dflash_llm, "base": base_llm}
        else:
            raise RuntimeError("PREFER_LOCAL_MODELS=False — use remote path")
    except RuntimeError as _local_e:
        print(_local_e)
        print("[LOCAL-LOAD] Falling back to remote model resolution...")
        model_name, backend = pick_model_name()
        vllm_llm, tokenizer, hf_model = load_models(model_name)

    # 4) Sanity: force the banner again so it is unmistakable in the log
    print_one_million_brains_banner(_LIVE_PATCH_SUCCESS)

    verify_inference_engine(vllm_llm, tokenizer)

    # 5) Evaluation / benchmark
    results: Dict[str, Any] = {}
    run_arc_eval = (
        not args.demo_only
        and args.eval_challenges
        and args.eval_solutions
    )
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
            "[FINAL][demo] Classic TPS: %.2f | MILLION-BRAINS TPS: %.2f | reallocs: %d | Avg accept: %.2f"
            % (
                demo["classic"]["tps"],
                demo["mbr"]["tps"],
                demo["mbr"].get("feature_reallocations", 0),
                demo["mbr"]["avg_accepted_per_block"],
            )
        )
    if "arc" in results:
        arc = results["arc"]
        print(
            "[FINAL][arc] Classic acc: %.2f%% | MBR acc: %.2f%% | tests: %d"
            % (
                arc["classic"]["accuracy"] * 100,
                arc["mbr"]["accuracy"] * 100,
                arc["classic"]["tests"],
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
            "k": K,
            "block_size": BLOCK_SIZE,
        }
        if "demo" in results:
            payload["demo"] = {
                "classic": {
                    k: v
                    for k, v in results["demo"]["classic"].items()
                    if k != "final_text"
                },
                "mbr": {
                    k: v
                    for k, v in results["demo"]["mbr"].items()
                    if k != "final_text"
                },
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
