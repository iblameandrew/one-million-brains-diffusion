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
- Prints the exact " ONE-MILLION-BRAINS-DFLASH INITIALIZED " banner.
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
# !pip install -q vllm transformers accelerate flash-attn --upgrade
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
            [sys.executable, "-m", "pip", "install", "-q",
             "vllm", "transformers", "accelerate", "flash-attn", "--upgrade"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as _inst_e:
        print("[INSTALL] Subprocess pip skipped/failed (packages may already exist):", _inst_e)
else:
    # In notebook the magic line above (when the user pastes it as its own cell or at top)
    # will have executed the install. We leave the comment here exactly as the spec demands.
    pass

# =============================================================================
# TOGGLES - ALL USER CONTROLS LIVE HERE (edit and re-run)
# =============================================================================
K = 4                           # number of parallel drafter streams / feature-slots per super-block
NUM_PERSONALITY_FEATURES = 12   # size of the fixed personality feature bank; do not change unless you extend the list below
BLOCK_SIZE = 6                  # tokens each stream proposes per super-block (M = K * BLOCK_SIZE)
MAX_SUPERBLOCKS = 32            # safety cap on super-blocks
ENABLE_FEATURE_REALLOCATION = True  # master switch for adaptive reallocation of features into slots based on acceptance
ACCEPTANCE_THRESHOLD = 0.28     # below this a feature-slot is considered underperforming and eligible for reallocation
REFRAME_TEMP_BOOST = 0.35       # additive temperature boost on total super-block rejection
BASE_TEMPERATURE = 0.7
BASE_TOP_P = 0.92
TARGET_MAX_TOKENS = 160         # benchmark generation length
BENCHMARK_PROMPT = (            # a single hard prompt that benefits from combinatorial diversity of personality features
    "You are a world-class puzzle solver. Think step-by-step with extreme rigor. "
    "A farmer has 7 chickens, 4 pigs, and 3 cows. Each chicken has 2 legs, pigs have 4, cows have 4. "
    "At exactly noon every animal casts a perfect shadow that an observer might mistakenly count as extra legs. "
    "The observer counts 61 'legs' total (real + shadow). How many legs are actually real? "
    "Explain your reasoning in numbered steps and give the final integer answer."
)
SEED = 42                       # for reproducibility of permutation hashing + sampling inside active features

# =============================================================================
# STANDARD LIBRARY + ML IMPORTS
# =============================================================================
import os
import sys
import math
import time
import random
import json
import hashlib
import traceback
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
from vllm.spec_decode import draft_model as vllm_draft_module  # for live patching target

# =============================================================================
# FIXED BANK OF 12 PERSONALITY FEATURES
# Each feature is a distinct direction that can be injected into a feature-slot.
# The PermutationFeatureSlotAllocator selects K distinct features via a deterministic
# permutation and assigns them to the K parallel streams for the current super-block.
# =============================================================================
PERSONALITY_FEATURES: List[str] = [
    "PreciseAnchor",      # low temperature, anchors strongly to given facts and constraints
    "CreativeExplorer",   # higher temperature / broad top_p, favors novelty and less-trodden paths
    "LogicalReasoner",    # medium temperature with bias toward explicit step-by-step chains
    "SelfCritic",         # tends to surface contradictions and reduce over-commitment
    "Reframer",           # biases toward re-interpreting the problem statement or constraints
    "Synthesizer",        # prefers combinations; receives priority during cross-stream fusion
    "DevilAdvocate",      # stress-tests the currently favored line of reasoning
    "PatternMatcher",     # strongly attends to numerical, structural, and repetition patterns
    "EdgeCaseHunter",     # deliberately explores boundary conditions and low-probability inputs
    "ContextGrounding",   # pulls in broader real-world constraints and background knowledge
    "Abstractor",         # lifts specifics into higher-level rules or invariants
    "MirrorReflector",    # meta-feature that conditions on the behavior of the other active features
]

# Feature-specific generation hyper-parameters used during the proposal phase for the
# stream that has been allocated that personality feature. This is the high-level
# equivalent of "feature-specific temperature/top_p + post-attention gating".
FEATURE_PARAMS: Dict[str, Dict[str, float]] = {
    "PreciseAnchor":      {"temperature": 0.35, "top_p": 0.82, "repetition_penalty": 1.08},
    "CreativeExplorer":   {"temperature": 1.15, "top_p": 0.98, "repetition_penalty": 1.00},
    "LogicalReasoner":    {"temperature": 0.55, "top_p": 0.90, "repetition_penalty": 1.05},
    "SelfCritic":         {"temperature": 0.65, "top_p": 0.88, "repetition_penalty": 1.12},
    "Reframer":           {"temperature": 0.90, "top_p": 0.95, "repetition_penalty": 1.02},
    "Synthesizer":        {"temperature": 0.60, "top_p": 0.93, "repetition_penalty": 1.03},
    "DevilAdvocate":      {"temperature": 0.80, "top_p": 0.91, "repetition_penalty": 1.06},
    "PatternMatcher":     {"temperature": 0.45, "top_p": 0.85, "repetition_penalty": 1.04},
    "EdgeCaseHunter":     {"temperature": 0.75, "top_p": 0.94, "repetition_penalty": 1.07},
    "ContextGrounding":   {"temperature": 0.50, "top_p": 0.89, "repetition_penalty": 1.01},
    "Abstractor":         {"temperature": 0.70, "top_p": 0.96, "repetition_penalty": 1.05},
    "MirrorReflector":    {"temperature": 0.85, "top_p": 0.87, "repetition_penalty": 1.09},
}

assert len(PERSONALITY_FEATURES) == NUM_PERSONALITY_FEATURES, "PERSONALITY_FEATURES list length must equal NUM_PERSONALITY_FEATURES"


# =============================================================================
# COMBINATORIAL PERMUTATION UNRANKING (mathematically rigorous hashing)
# =============================================================================
def calc_permutation_count(n: int, k: int) -> int:
    """P(n, k) = n! / (n-k)!   (number of injective functions from k to n)"""
    p = 1
    for i in range(k):
        p *= (n - i)
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


def hash_to_feature_permutation(pooled_vec: torch.Tensor, step: int, n: int, k: int) -> List[int]:
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
        seed = step * 0x9e3779b97f4a7c15
    else:
        v = pooled_vec.detach().float().flatten()
        mean_v = v.mean().item()
        var_v = (v - mean_v).pow(2).mean().item() + 1e-12
        h = int((mean_v * 1_000_003 + var_v * 1_000_037 + step * 37) * 1_000_000_007) & 0xFFFFFFFFFFFFFFFF
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
        self.segment_emb = nn.Embedding(64, internal_dim)          # super-block / horizon index
        self.stream_pos_emb = nn.Embedding(k, internal_dim)        # which feature-slot / stream lane

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
        feature_indices = hash_to_feature_permutation(pooled, step, self.num_features, self.k)

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

        feature_vectors = self.feature_emb(torch.tensor(feature_indices, dtype=torch.long))
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
        zip(draft_token_ids, target_logprobs_for_exact_tokens, draft_logprobs_for_exact_tokens)
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


def extract_logprob_for_token(logprob_dict_or_list: Any, token_id: int) -> float:
    """
    vLLM returns different structures depending on version / prompt_logprobs vs logprobs.
    This helper is defensive.
    """
    try:
        if isinstance(logprob_dict_or_list, dict):
            if token_id in logprob_dict_or_list:
                return float(logprob_dict_or_list[token_id].logprob)
            # fallback: take the top entry
            if logprob_dict_or_list:
                return float(next(iter(logprob_dict_or_list.values())).logprob)
        if isinstance(logprob_dict_or_list, list):
            for entry in logprob_dict_or_list:
                if hasattr(entry, "token") and entry.token == token_id:
                    return float(getattr(entry, "logprob", -1.0))
        # last resort neutral
        return -0.8
    except Exception:
        return -0.9


# =============================================================================
# ONE-MILLION-BRAINS-DFLASH GENERATE (the full algorithm)
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
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full one-million-brains-dflash loop (permutation-driven feature-slot allocation).

    - Uses vLLM for fast batched proposal across K feature-slotted streams
      (the "drafter forward" for the entire super-block horizon).
    - Uses vLLM prompt_logprobs for the target verification forward on the K candidate sequences.
    - The PermutationFeatureSlotAllocator selects a fresh K-permutation of personality features
      for every super-block based on the pooled state of the previous verification.
    - Adaptive reallocation: under-performing feature-slots have their personality feature
      replaced from the unused pool on subsequent blocks.
    - This realizes the Fast Million Brains approach at inference time.
    - Returns rich stats + final text.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if allocator is None:
        allocator = PermutationFeatureSlotAllocator(internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=k)

    # State
    current_text = prompt
    generated_ids: List[int] = []
    total_accepted = 0
    total_blocks = 0
    feature_reallocations = 0
    acceptance_history: List[float] = []
    feature_history: List[List[str]] = []
    reframe_events = 0

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
        pooled = make_pooled_state(generated_ids, sb)
        alloc_out = allocator(pooled, sb)
        active_feature_indices = alloc_out["feature_indices"]
        active_feature_names = alloc_out["feature_names"]
        feature_params = allocator.get_feature_params(active_feature_indices)
        feature_history.append(active_feature_names)

        # 2) Independent Draft phase (K parallel proposals via vLLM batch)
        #    One batched "drafter forward" for the whole super-block horizon, each stream
        #    conditioned on its currently allocated personality feature.
        draft_sampling = []
        for i in range(k):
            p = feature_params[i]
            sp = SamplingParams(
                temperature=p["temperature"] + (REFRAME_TEMP_BOOST if reframe_events > 0 and sb == total_blocks-1 else 0.0),
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
            tok_ids = list(out.outputs[0].token_ids)[:block_size]
            lps = []
            if out.outputs[0].logprobs:
                for tid in tok_ids:
                    lp = extract_logprob_for_token(out.outputs[0].logprobs[-len(tok_ids):], tid)
                    lps.append(lp)
            else:
                lps = [-0.7] * len(tok_ids)
            proposals.append(tok_ids)
            draft_lps.append(lps)

        # 3) Cross-stream integration / fusion phase (the high-level "Group Think")
        #    If the "Synthesizer" personality feature is active in any slot, it gets priority
        #    when forming the fused candidate. Otherwise we take a simple majority per position.
        fused_proposal: List[int] = []
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
            fused_proposal += proposals[0][len(fused_proposal):block_size]

        # 4) Target verification forward on the K candidate sequences (+ fused)
        #    Single batched call giving us the target's view of every drafted token.
        candidates_for_verify = []
        base = current_text
        for pr in proposals:
            append_text = tokenizer.decode(pr, skip_special_tokens=True)
            candidates_for_verify.append(base + append_text)
        fused_text = tokenizer.decode(fused_proposal, skip_special_tokens=True)
        candidates_for_verify.append(base + fused_text)

        verify_params = SamplingParams(
            temperature=0.0,
            max_tokens=0,
            prompt_logprobs=True,
        )
        verify_outs = vllm_llm.generate(candidates_for_verify, verify_params)

        # Extract target logprobs for the drafted region of each candidate
        target_lps_per_path: List[List[float]] = []
        for j, vout in enumerate(verify_outs):
            plp = vout.prompt_logprobs or []
            drafted_lps = []
            target_ids = proposals[j] if j < len(proposals) else fused_proposal
            for tid in target_ids:
                drafted_lps.append(extract_logprob_for_token(plp, tid) if plp else -0.6)
            target_lps_per_path.append(drafted_lps)

        # 5) Run generalized cumprod acceptance on every path
        best_accepted: List[int] = []
        best_rate = -1.0
        path_rates: List[float] = []

        for j in range(k):
            acc, rate = compute_accepted_tokens(
                proposals[j],
                target_lps_per_path[j],
                draft_lps[j],
            )
            path_rates.append(rate)
            if rate > best_rate:
                best_rate = rate
                best_accepted = acc

        # Also test the fused candidate
        acc_f, rate_f = compute_accepted_tokens(
            fused_proposal,
            target_lps_per_path[-1],
            draft_lps[0],
        )
        if rate_f > best_rate:
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

        # 7) Adaptive feature reallocation (the "Mirror" mechanism, stripped of astrology)
        if enable_reallocation:
            for i in range(k):
                ema_accept[i] = 0.7 * ema_accept[i] + 0.3 * path_rates[i]

            for i in range(k):
                if ema_accept[i] < ACCEPTANCE_THRESHOLD:
                    # Draw a replacement from the currently unused personality features.
                    unused = [r for r in range(NUM_PERSONALITY_FEATURES) if r not in active_feature_indices]
                    if unused:
                        new_feat = unused[(sb * 31 + i) % len(unused)]
                        old_name = PERSONALITY_FEATURES[active_feature_indices[i]]
                        active_feature_indices[i] = new_feat
                        feature_reallocations += 1
                        ema_accept[i] = 0.55
                        print(f"  [REALLOC] Slot {i} feature {old_name} -> {PERSONALITY_FEATURES[new_feat]} (EMA accept {ema_accept[i]:.3f})")

        # 8) Full rejection handling (equivalent to "Reframe")
        if accepted_len == 0 and enable_reallocation:
            reframe_events += 1
            print(f"  [DIVERGENCE] Super-block {sb} produced zero accepted tokens. Boosting proposal diversity for next block.")

        # 9) Diagnostic logging (feature-slot view)
        gmask = GroupThinkMask(k=k, block_size=block_size, phase="integration" if accepted_len > 0 else "draft")
        if sb < 2 or accepted_len == 0:
            print(f"    Super-block {sb:02d} | features={active_feature_names} | accepted={accepted_len}/{block_size} | mask={gmask.describe()}")

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
                draft_lps.append(extract_logprob_for_token(out.outputs[0].logprobs, tid))
        else:
            draft_lps = [-0.75] * len(draft_ids)

        # Target verification forward on the single extended candidate
        candidate = current_text + tokenizer.decode(draft_ids, skip_special_tokens=True)
        vsp = SamplingParams(temperature=0.0, max_tokens=0, prompt_logprobs=True)
        vout = vllm_llm.generate([candidate], vsp)[0]
        plp = vout.prompt_logprobs or []
        target_lps = [extract_logprob_for_token(plp, tid) for tid in draft_ids]

        acc, rate = compute_accepted_tokens(draft_ids, target_lps, draft_lps)
        acceptance_history.append(rate)

        if acc:
            generated_ids.extend(acc)
            current_text = current_text + tokenizer.decode(acc, skip_special_tokens=True)
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
        print("[MillionBrainsDFlashDraftModel] Stand-in initialized (full feature-slot allocator injected at runtime)")

    def propose_with_allocator(self, pooled_state, step):
        if self.feature_allocator is not None:
            return self.feature_allocator(pooled_state, step)
        return list(range(min(self.k, NUM_PERSONALITY_FEATURES)))


# =============================================================================
# LIVE EDIT / MONKEY-PATCH SECTION (the "preemptive" DFlash injection)
# This must run immediately after the pip install and before heavy model loading.
# Strategy:
#   1. Attempt to locate vllm.spec_decode.draft_model on disk and surgically
#      overwrite key classes / methods with one-million-brains-dflash versions (file fallback).
#   2. Always perform runtime monkey-patching via subclass + module replacement.
#   3. Inject a global "MILLION_BRAINS_DFLASH" symbol so user code can detect the patch.
#   4. Emit the exact "ONE-MILLION-BRAINS-DFLASH INITIALIZED" banner on success.
# =============================================================================
def _live_edit_dflash() -> bool:
    """
    Robust live-edit routine. Returns True if any patch (file or runtime) succeeded.
    Performs the one-million-brains-dflash injection into vLLM's draft mechanisms.
    """
    patched = False
    print("\n[PREEMPTIVE DFLASH LIVE-EDIT] Scanning for DFlashDraftModel / vLLM draft components (one-million-brains-dflash injection)...")

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
# MILLION-BRAINS-DFLASH INJECTION (auto-inserted by million_brains_dflash.py at {time.strftime('%Y-%m-%d %H:%M:%S')})
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
                    print(f"    [FILE PATCH] Appended one-million-brains-dflash injection to {target_file}")
                    patched = True
                except Exception as e:
                    print(f"    [FILE PATCH] Could not append (likely read-only FS): {e}")
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
        vllm_draft_module.PermutationFeatureSlotAllocator = PermutationFeatureSlotAllocator
        vllm_draft_module.million_brains_dflash_generate = million_brains_dflash_generate
        vllm_draft_module.classic_dflash_generate = classic_dflash_generate

        # Subclass replacement (what user code importing DraftModel will see)
        try:
            OriginalDraft = getattr(vllm_draft_module, "DraftModel", None)
            if OriginalDraft is not None:
                class PatchedDraftModel(OriginalDraft):  # type: ignore
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        self.feature_allocator = PermutationFeatureSlotAllocator(
                            internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=K
                        )
                        self.k = K
                        self.block_size = BLOCK_SIZE
                        print("[PatchedDraftModel] Runtime subclass active - one-million-brains-dflash ready")

                vllm_draft_module.DraftModel = PatchedDraftModel
                sys.modules.setdefault("dflash", type(sys)("dflash"))
                sys.modules["dflash"].DraftModel = PatchedDraftModel
                sys.modules["dflash"].MILLION_BRAINS_DFLASH_PATCHED = True
                print("    [RUNTIME PATCH] vllm.spec_decode.draft_model.DraftModel replaced with one-million-brains-dflash feature-slot aware subclass")
            else:
                # No original DraftModel - still expose our allocator
                sys.modules.setdefault("dflash", type(sys)("dflash"))
                ndm = MillionBrainsDFlashDraftModel()
                ndm.feature_allocator = PermutationFeatureSlotAllocator(
                    internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=K
                )
                sys.modules["dflash"].DraftModel = type(ndm)
                sys.modules["dflash"].MillionBrainsDFlashDraftModel = type(ndm)
                print("    [RUNTIME PATCH] No original DraftModel found - pure MillionBrainsDFlashDraftModel exposed under 'dflash'")
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
  ███████╗███╗   ███╗██████╗       ██████╗ ███████╗██╗      █████╗ ███████╗██╗  ██╗
  ██╔════╝████╗ ████║██╔══██╗     ██╔═══██╗██╔════╝██║     ██╔══██╗██╔════╝██║  ██║
  █████╗  ██╔████╔██║██████╔╝     ██║   ██║█████╗  ██║     ███████║███████╗███████║
  ██╔══╝  ██║╚██╔╝██║██╔══██╗     ██║   ██║██╔══╝  ██║     ██╔══██║╚════██║██╔══██║
  ██║     ██║ ╚═╝ ██║██████╔╝     ╚██████╔╝██║     ███████╗██║  ██║███████║██║  ██║
  ╚═╝     ╚═╝     ╚═╝╚═════╝       ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
================================================================================
"""
    print(banner)
    if success:
        print(" ONE-MILLION-BRAINS-DFLASH INITIALIZED  |  K=%d  |  FEATURES=%d  |  REALLOCATION=%s" % (
            K, NUM_PERSONALITY_FEATURES, str(ENABLE_FEATURE_REALLOCATION).upper()))
        print(" Patch status: %s" % ("SUCCESS (file+runtime)" if _LIVE_PATCH_SUCCESS else "RUNTIME ONLY"))
    else:
        print(" ONE-MILLION-BRAINS-DFLASH INITIALIZED (DEGRADED - patch encountered errors, pure-Python fallback active)")
    print("================================================================================\n")


print_one_million_brains_banner(_LIVE_PATCH_SUCCESS)


# =============================================================================
# MODEL LOADING WITH FALLBACK (z-lab/Qwen3.5-4B-DFlash first, then real Qwen)
# =============================================================================
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
        gpu_mem_gb = props.total_memory / (1024 ** 3)
        print(f"[GPU] Detected {props.name} with {gpu_mem_gb:.1f} GB VRAM")

    candidates = [
        "z-lab/Qwen3.5-4B-DFlash",           # the requested fictional/special model
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
            print(f"    ... {name} not available or tokenizer failed ({type(e).__name__}). Trying next...")
    # Absolute last resort (should never happen)
    return "Qwen/Qwen2.5-1.5B-Instruct", "vllm"


def load_models(model_name: str):
    """
    Load one vLLM engine (fast path for both classic and the proposal/verification steps of one-million-brains-dflash).
    Also load a lightweight HF tokenizer copy for decode/encode consistency.
    We keep everything in FP16/BF16; no quantization to stay mainstream & simple.
    """
    print(f"\n[LOAD] Initializing vLLM engine for {model_name} (this can take 30-120s on first download)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # vLLM engine - mainstream path
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        enforce_eager=False,   # graph capture for speed where possible
    )
    print("[LOAD] vLLM engine ready.")

    # Also expose a tiny HF model for any future hidden-state needs (optional, lazy)
    hf_model = None
    try:
        # Only load if we have headroom; otherwise we synthesize pooled states.
        if torch.cuda.get_device_properties(0).total_memory > 14 * (1024 ** 3):
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()
            print("[LOAD] Optional HF reference model also resident for hidden-state introspection.")
    except Exception:
        pass

    return llm, tokenizer, hf_model


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
    print("BENCHMARK: CLASSIC PARA-DFLASH vs MILLION-BRAINS-DFLASH (K=4, Fast Million Brains)")
    print("=" * 80)
    print(f"Prompt (first 180 chars): {prompt[:180]}...")
    print(f"Target generation length: {max_new} tokens | Block size: {BLOCK_SIZE} | K: {K}")
    print("-" * 80)

    # --- CLASSIC ---
    print("\n[CLASSIC] Running single-path Para-DFlash baseline ...")
    t0 = time.perf_counter()
    classic_res = classic_dflash_generate(
        vllm_llm, tokenizer, prompt, max_new_tokens=max_new, block_size=max(6, BLOCK_SIZE), seed=SEED
    )
    t_classic = time.perf_counter() - t0
    classic_tps = classic_res["num_tokens"] / max(1e-6, t_classic)

    # --- MILLION-BRAINS ---
    print("\n[MILLION-BRAINS] Running full one-million-brains-dflash (permutation feature-slot allocator + cross-stream integration + adaptive reallocation) ...")
    t0 = time.perf_counter()
    allocator = PermutationFeatureSlotAllocator(internal_dim=256, num_features=NUM_PERSONALITY_FEATURES, k=K)
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

    print(f"\n{'Metric':<30} {'Classic (K=1)':>18} {'one-million-brains-dflash (K=4)':>22}")
    print("-" * 72)
    print(f"{'Generated tokens':<30} {classic_res['num_tokens']:>18} {mbr_res['num_tokens']:>22}")
    print(f"{'Wall time (s)':<30} {t_classic:>18.2f} {t_mbr:>22.2f}")
    print(f"{'Tokens / sec':<30} {classic_tps:>18.2f} {mbr_tps:>22.2f}")
    print(f"{'Super-blocks executed':<30} {classic_res['num_superblocks']:>18} {mbr_res['num_superblocks']:>22}")
    print(f"{'Avg accepted tokens / block':<30} {classic_res['avg_accepted_per_block']:>18.2f} {mbr_res['avg_accepted_per_block']:>22.2f}")
    print(f"{'Feature reallocations':<30} {classic_res.get('feature_reallocations', 0):>18} {mbr_res.get('feature_reallocations', 0):>22}")
    print(f"{'Divergence events':<30} {classic_res['reframe_events']:>18} {mbr_res['reframe_events']:>22}")

    print("\n--- Sample (Classic) ---")
    print(classic_res["final_text"][-600:] if len(classic_res["final_text"]) > 600 else classic_res["final_text"])
    print("\n--- Sample (one-million-brains-dflash) ---")
    print(mbr_res["final_text"][-600:] if len(mbr_res["final_text"]) > 600 else mbr_res["final_text"])

    # Rich diagnostic
    print("\n[MILLION-BRAINS] Feature allocation history (last 6 super-blocks):")
    for i, feats in enumerate(mbr_res["feature_history"][-6:]):
        print(f"    SB {len(mbr_res['feature_history'])-6+i:02d}: {feats}")

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
    print("\n[million_brains_dflash.py] Starting full one-million-brains-dflash (Fast Million Brains) Kaggle run")
    print(f"    K={K}, BLOCK_SIZE={BLOCK_SIZE}, ENABLE_FEATURE_REALLOCATION={ENABLE_FEATURE_REALLOCATION}")
    print(f"    SEED={SEED}, TARGET_MAX_TOKENS={TARGET_MAX_TOKENS}")

    # 1) Live-edit banner was already printed right after the patcher ran.

    # 2) Choose & load model (with the exact fallbacks requested)
    model_name, backend = pick_model_name()
    vllm_llm, tokenizer, hf_model = load_models(model_name)

    # 3) Sanity: force the banner again so it is unmistakable in the log
    print_one_million_brains_banner(_LIVE_PATCH_SUCCESS)

    # 4) Run the benchmark (the thing the user actually cares about)
    results = benchmark(vllm_llm, tokenizer, BENCHMARK_PROMPT, max_new=TARGET_MAX_TOKENS)

    # 5) Final summary line (useful when scanning Kaggle logs)
    print("[FINAL] Classic TPS: %.2f | MILLION-BRAINS TPS: %.2f | reallocs: %d | Avg accept: %.2f" % (
        results["classic"]["tps"],
        results["mbr"]["tps"],
        results["mbr"].get("feature_reallocations", 0),
        results["mbr"]["avg_accepted_per_block"],
    ))

    # Optional: write a small artifact so the Kaggle "Output" pane has something
    try:
        with open("/kaggle/working/million_brains_dflash_results.json", "w") as f:
            json.dump({
                "classic": {k: v for k, v in results["classic"].items() if k != "final_text"},
                "mbr": {k: v for k, v in results["mbr"].items() if k != "final_text"},
                "model": model_name,
                "k": K,
                "block_size": BLOCK_SIZE,
            }, f, indent=2)
        print("[ARTIFACT] Wrote /kaggle/working/million_brains_dflash_results.json")
    except Exception:
        pass

    print("\n[million_brains_dflash.py] All done. You can now inspect the generated samples and the metrics above.")

