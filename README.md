# one-million-brains-diffusiongemma

<img width="784" height="1168" alt="image" src="https://github.com/user-attachments/assets/41dd8f5d-7fb1-4437-acd5-ddc00146e6d9" />

**Permutation-Gated Feature-Slot Diffusion** on DiffusionGemma — HuggingFace block-diffusion with a Million-Brains orchestration layer (prompt + sampling conditioning, not draft-model speculative decoding).

Single Kaggle script: `million_brains_dflash.py`. `SCRIPT_VERSION = 2026-06-21-diffusion-d-hf`. HuggingFace-only (`DiffusionGemmaForBlockDiffusion`); no vLLM. Wanted to see what happened if i live-wired a QNN from my open-deepthink problem into a diffusion superblock.

**DISCLAIMER: CURRENT PERFORMANCE OF THIS TECHNIQUE IN ARC-AGI-2 IS 2/120 - SO IM ADDING THIS TO MY COLLECTION OF TOYS**

## What it is

| Component | Role |
|-----------|------|
| **DiffusionGemma (HF)** | Block-diffusion engine: 256-token canvas, iterative denoise + commit |
| **PermutationFeatureSlotAllocator** | Hash pooled state → permutation of K features from 12 spatial primitives |
| **CTSB circuit smoothing** | Limits primitive-slot swaps between denoise super-blocks |
| **Cross-stream fusion + cumprod verify** | Accept/reject parallel trajectories before canvas commit |
| **ARC spatial ensemble** | Phase 1: 8 primitive-conditioned JSON grids; Phase 2: pixel majority vote |

Conditioning is **prompt + greedy sampling params** only. No hidden-state injection, no fiber embeddings, no cohomological stitch in the default path (`ENABLE_TFBD = False`).

## Two execution pipelines

| Pipeline | When | Technique |
|----------|------|-----------|
| **ARC eval** (default with data mounted) | Competition/local ARC JSON present | Two-phase **spatial grid ensemble** — no Million-Brains denoise loop |
| **Demo benchmark** | `--demo-only` or `ARC_DATA_PROFILE=off` | **K=4** parallel conditioned trajectories per denoise super-block |

`K=4` (benchmark) and `ARC_HYPOTHESIS_SLOTS=8` (ARC Phase 1) are independent.

## ARC eval technique (default path)

When competition data is attached, the script evaluates ARC-AGI **without** the benchmark denoise loop:

**Phase 1 — spatial hypothesis pool** (`ARC_HYPOTHESIS_SLOTS = 8`)

1. `PermutationFeatureSlotAllocator` permutes 8 primitives from the 12-lens bank.
2. Per slot: spatial-lens prompt (train pairs + test input + primitive instruction).
3. HF `generate()` calls (`ARC_PHASE1_PROMPT_PARALLELISM` controls batching vs sequential).
4. `ARC_SPATIAL_ENABLE_THINKING = False` — greedy JSON decode, `[[` prefill, thinking off.
5. Parse each response into a 2D integer grid.

**Phase 2 — pixel majority vote** (`ARC_SPATIAL_GRID_ENSEMBLE = True`)

- Per-cell plurality across parsed Phase-1 grids → final output grid.
- **No LLM synthesis**; Phase 2 is deterministic voting only.

Outputs: PNG grade cards (`arc_grades/`), `tfbd_results.json`, accuracy summary.

## Demo benchmark technique (Million-Brains denoise)

Only when running `--demo-only` or when ARC paths are unavailable:

At each denoise super-block (`DIFFUSION_DENOISE_CHUNK = 6` tokens):

1. Hash history → permutation of **K=4** features from 12 primitives.
2. **CTSB** circuit smoothing limits slot swaps between super-blocks.
3. **K** parallel `generate()` calls with per-slot lens prompts.
4. Cross-stream fusion + cumprod logprob verification + adaptive reallocation.
5. Accepted tokens commit into the 256-token block-diffusion canvas; loop continues.

## The 12 spatial primitives

`Rotate90`, `Rotate180`, `ReflectH`, `ReflectV`, `Transpose`, `CropBBox`, `TileRepeat`, `ColorMap`, `SymmetryComplete`, `FloodFill`, `ComponentExtract`, `GravityShift` — each has a per-slot prompt lens in `SPATIAL_PRIMITIVE_LENSES`.

## Key files

| Path | Purpose |
|------|---------|
| `million_brains_dflash.py` | Main entry (load, verify, ARC eval, benchmark, CLI) |
| `tfbd.py` | Backward-compatible alias → runs `million_brains_dflash.py` |
| `agent-tools/verify_arc_phase1.py` | CPU-only structure / budget tests |
| `agent-tools/test_pixel_vote.py` | Pixel vote unit tests |

## Quick start (Kaggle)

### 1. Notebook inputs

| Input | Handle |
|-------|--------|
| Model | `google/diffusiongemma` → `diffusiongemma-26b-a4b-it` |
| Competition | `arc-prize-2026-arc-agi-2` |
| Wheels (offline) | `godelcomplete/vllm-gemma` → `transformers_latest_wheels/` |

### 2. Dependencies (Cell 1 — restart kernel after)

```python
!pip install --force-reinstall --no-index \
    --find-links=/kaggle/input/notebooks/godelcomplete/vllm-gemma/transformers_latest_wheels/ \
    transformers==5.12.1
!pip install -q "accelerate>=0.26.0" "safetensors>=0.4.0"
```

No bitsandbytes. Loading uses an explicit per-layer `device_map` across your GPUs.

### 3. Run

```python
!python million_brains_dflash.py --arc-profile auto --arc-split evaluation
```

Expected banner: `ONE-MILLION-BRAINS-DIFFUSIONGEMMA INITIALIZED`.

Benchmark only: `!python million_brains_dflash.py --demo-only`

### 4. Hardware

- **26B MoE** uses manual per-layer `device_map` (avoids accelerate tie-weight crashes on DiffusionGemma).
- **4×22GB** works with batched/sequential Phase 1; **A100 80GB** is more comfortable.

## Configuration

All toggles live in the `TOGGLES` block at the top of `million_brains_dflash.py`:

- `ENABLE_TFBD` — `False` by default; set `True` only for experimental fiber-bundle path (not the documented technique)
- `ALLOCATOR_MODE` — `"permutation"` by default (`"fiber"` / `"hybrid"` are TFBD experimental)
- `ARC_DATA_PROFILE` — `"auto"` | `"kaggle"` | `"local"` | `"off"`
- `ARC_HYPOTHESIS_SLOTS` — Phase-1 pool size (default 8)
- `ARC_SPATIAL_GRID_ENSEMBLE` — grid hypotheses + pixel majority Phase 2
- `EVAL_MAX_TASKS` / `EVAL_SMOKE_TASK_ID` — smoke-test scope
- `K` — demo benchmark parallel trajectories only

## Reading the logs

| Prefix | Meaning |
|--------|---------|
| `[MBR-DIFFUSION]` / `[DIFFUSION]` | Benchmark denoise loop |
| `[ARC-PHASE-1]` | Spatial hypothesis generation |
| `[ARC-PHASE-2]` | Pixel majority vote across parsed grids |
| `[FINAL][arc]` | Dataset accuracy |

## Verification

```bash
python agent-tools/verify_arc_phase1.py
python agent-tools/test_pixel_vote.py
```

## License / usage

Educational and research prototype. Pull requests welcome.
