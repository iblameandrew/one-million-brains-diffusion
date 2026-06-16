# million-brains-dflash

<img width="1920" height="2400" alt="image" src="https://github.com/user-attachments/assets/3d47988b-261a-479e-bb1f-f074dfadefc2" />


**Permutation-Gated Feature-Slot Speculative Decoding** — the Fast Million Brains approach.

A complete, self-contained, runnable Kaggle script (`million_brains_dflash.py`) that implements an advanced, diversity-driven speculative decoding architecture on top of vLLM.

## What it is

Instead of a single linear draft, this system maintains a small fixed bank of **12 personality features** (cognitive biases / thinking styles):

- PreciseAnchor, CreativeExplorer, LogicalReasoner, SelfCritic, Reframer, Synthesizer, DevilAdvocate, PatternMatcher, EdgeCaseHunter, ContextGrounding, Abstractor, MirrorReflector

At every super-block it uses a **deterministic permutation grammar** (combinadic unranking of P(12, K)) to select and assign K distinct features to K parallel "feature-slots". Each slot runs with its own feature embedding bias, positional offset, and sampling hyperparameters.

A lightweight two-phase process (independent drafting → cross-stream fusion) + target verification + generalized cumprod acceptance allows the system to accept variable-length coherent continuations while exploring multiple reasoning trajectories in parallel.

Adaptive reallocation ("mirror descent") swaps under-performing features out of slots on the fly.

This is the **Fast Million Brains** idea: rapidly reconfiguring a compact set of high-signal cognitive circuits into thousands of different parallel thought configurations every few tokens, rather than relying on a single sequential stream.

## Key Files

- `million_brains_dflash.py` — the full implementation (installs vLLM + dependencies, live monkey-patches for DFlash-style extension, runs a benchmark comparing classic vs. million-brains-dflash mode).

## Quick Start (Kaggle)

1. Upload `million_brains_dflash.py` as a Script or Notebook.
2. Run it. It will:
   - `!pip install ...` (or equivalent)
   - Perform the live-edit / monkey-patch of vLLM draft mechanisms
   - Print the **MILLION-BRAINS-DFLASH INITIALIZED** banner
   - Load a small Qwen model (with fallbacks)
   - Run head-to-head benchmark (classic vs. full feature-slot parallel mode)
   - Report tokens/sec, average accepted tokens per block, feature reallocations, etc.

## How the "Grammar" Works

The core combinatorial engine lives in `PermutationFeatureSlotAllocator`:

- `hash_to_feature_permutation(pooled_vec, step, n=12, k=4)` — turns previous verification hidden state into a stable seed → rank → ordered K-tuple via factorial number system unranking.
- This produces a valid injection: which 4 features go into which of the K slots for the next super-block.
- Feature vectors, stream-positional embeddings, and per-feature sampling params are then applied to the K parallel streams.
- After drafting + fusion + verification, the new pooled state feeds the next permutation.

All 11,880 possible allocations (for K=4) are reachable and deterministic given history. The "circuits" (feature embeddings, gates, sampling biases) are fixed atoms; the grammar composes them into new temporary parallel reasoning circuits on every super-block.

## Current Implementation Notes

- High-level simulation using vLLM's `LLM.generate()` with batched heterogeneous requests (vLLM handles the actual multi-request batching).
- Feature injection and cross-stream "group think" are approximated via per-request SamplingParams + Python-side fusion.
- The low-level vision (true per-slot hierarchical attention masks, feature embeddings added inside the model forward, proper shared-prefix KV for divergent branches) is documented in the code comments and the `NexusDFlashDraftModel` / patcher sections for future deeper integration.

## Benchmark Output (example)

The script ends with a comparison table:

- Tokens/sec for classic vs. million-brains-dflash
- Average accepted tokens per super-block
- Number of feature reallocations (how often the permutation grammar + mirror mechanism swapped in better features)
- Side-by-side generated samples

## Requirements

- vLLM (mainstream recent version)
- transformers, accelerate, flash-attn
- CUDA GPU (tested conceptually on 4060m-class hardware; scales with better batching hardware)

## Theoretical Speedup

See earlier analysis in the thread. With a baseline of ~40 tps on a 4B model:

- Realistic: 1.6×–2.2× (64–88+ tps)
- At an assumed effective 300 tps system throughput: a hard math olympiad problem is estimated to take **~35–70 seconds** of wall time for a rigorous solution (highly dependent on actual acceptance rate `α` achieved by the parallel feature slots).

## Future Directions (not yet implemented)

- Lower-level vLLM integration (LLMEngine + custom model runner) for true prefix KV sharing and tree attention across branches.
- Actual injection of `feature_emb` / per-feature gating inside the transformer forward pass.
- Custom hierarchical attention masks matching the documented two-phase Group Think / non-causal intra-block + causal inter-block design.
- Proper training or distillation of the feature embedding tables.

## License / Usage

Educational + research prototype. The code is heavily commented for clarity around the permutation math, the feature-slot allocator, and the speculative acceptance logic.

Pull requests and experiments welcome — especially measurements of real acceptance rates on reasoning-heavy prompts and attempts at deeper vLLM patching.

---

Built as a thought experiment in "fast million brains" inference: a small number of high-quality fixed cognitive circuits + an extremely fast deterministic grammar that can reconfigure them into many parallel reasoning paths on every super-block.
