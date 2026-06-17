# one-million-brains-diffusiongemma


<img width="784" height="1168" alt="image" src="https://github.com/user-attachments/assets/41dd8f5d-7fb1-4437-acd5-ddc00146e6d9" />


**Permutation-Gated Feature-Slot Diffusion** — the Fast Million Brains approach on DiffusionGemma.

This repo wires a deterministic circuit grammar (12 spatial primitives, K parallel slots per denoise step) into DiffusionGemma block-diffusion via orchestration-layer conditioning — not draft-model speculative decoding.

## What it is

- **Engine:** DiffusionGemma (vLLM `diffusion_config` + entropy-bound denoising)
- **Controller:** `PermutationFeatureSlotAllocator` + CTSB circuit smoothing + adaptive reallocation
- **Per step:** K conditioned trajectories → cross-stream fusion → cumprod verification → commit into canvas
- **ARC eval:** spatial grid ensemble (Phase-1 hypotheses + pixel majority vote) with optional multi-agent voter pool

## Key files

- `million_brains_dflash.py` — full Kaggle script (installs vLLM, loads DiffusionGemma, runs benchmark + ARC eval)
- `mbr_voter_worker.py` — thin voter-pool worker entry

## Quick start (Kaggle)

1. Add Models input: `google/diffusiongemma` → `diffusiongemma-26b-a4b-it`
2. Add competition input: `arc-prize-2026-arc-agi-2` (for eval)
3. Paste/run `million_brains_dflash.py` (A100+ recommended for 26B)
4. Expect banner: `ONE-MILLION-BRAINS-DIFFUSIONGEMMA INITIALIZED`

## Circuit grammar (orchestration layer)

At each denoise super-block:

1. Hash pooled history → permutation of K features from 12 primitives
2. CTSB geodesic step limits circuit jumps; SPI blends sampling params
3. K parallel `vllm_llm.generate()` calls with per-slot lens prompts
4. Fusion + target verification + cumprod acceptance
5. Committed text feeds the next circuit selection

Feature injection is **prompt + sampling-params** level today (not hidden-state kernel patch).

## License / usage

Educational + research prototype. Pull requests welcome.