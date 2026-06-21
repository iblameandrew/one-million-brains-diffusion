# Topological Fiber-Bundle Diffusion (TFBD)

<img width="784" height="1168" alt="image" src="https://github.com/user-attachments/assets/41dd8f5d-7fb1-4437-acd5-ddc00146e6d9" />

**TFBD × DiffusionGemma** — ARC-AGI matrix completion with structured 2D fiber embeddings over HuggingFace block-diffusion.

Single Kaggle script: `tfbd.py` (`SCRIPT_VERSION = 2026-06-20-tfbd-u`). HuggingFace-only (`DiffusionGemmaForBlockDiffusion`); no vLLM, no draft-model speculative decoding.

## What it is

| Layer | Role |
|-------|------|
| **DiffusionGemma (HF)** | Block-diffusion engine: 256-token canvas, iterative denoise + commit |
| **TopologicalFiberEmbedding** | Per-cell `E = E_value + E_row + E_col + E_symmetry` |
| **CosmosSparsifier + TorusCache** | Latent sparsification + `T²` base-space projection |
| **FiberBundleDenoiser** | Partial re-masking: lock logic skeleton, explore fiber stalks |
| **CohomologicalStitcher** | K-trajectory copresheaf stitch + homology PRM proxy (β₀, β₁, χ) |
| **TFBD_Orchestrator** | Wraps DiffusionGemma; fiber KV bias + `inputs_embeds` injection |

Training-free orchestration only — no backward passes.

## Two execution pipelines

| Pipeline | When | Technique |
|----------|------|-----------|
| **ARC eval** (default with data mounted) | Competition/local ARC JSON present | Phase 1: 8 spatial-primitive JSON grids → Phase 2: stitch or vote |
| **Demo benchmark** | `--demo-only` or `ARC_DATA_PROFILE=off` | `K=4` TFBD fiber-bundle trajectories per denoise super-block |

`K=4` (benchmark) and `ARC_HYPOTHESIS_SLOTS=8` (ARC Phase 1) are independent.

## ARC eval technique (default: `ENABLE_TFBD=True`)

**Phase 1 — spatial hypothesis pool** (`ARC_HYPOTHESIS_SLOTS = 8`)

1. `FiberPrimitiveAllocator` (or permutation/hybrid via `ALLOCATOR_MODE`) picks 8 primitives from the 12-lens bank.
2. Per slot: spatial-lens prompt (train pairs + test input + primitive instruction).
3. Batched HF `generate()` when `ARC_PHASE1_PROMPT_PARALLELISM = True` (serial fallback for huge grids).
4. `ARC_SPATIAL_ENABLE_THINKING = False` — slots emit JSON grids only (`[[` prefill, greedy decode).
5. TFBD injects `TopologicalFiberEmbedding` from test input into `inputs_embeds` when `ENABLE_TFBD=True`.
6. Parse each response into a 2D integer grid.

**Phase 2 — grid fusion** (`ARC_SPATIAL_GRID_ENSEMBLE = True`)

| `ENABLE_TFBD` | Phase 2 method |
|---------------|----------------|
| `True` (default) | **CohomologicalStitcher** — PRM/Betti-proxy copresheaf row stitch across parsed grids |
| `False` | **Pixel majority vote** — per-cell plurality (no LLM synthesis) |

Outputs: PNG grade cards (`arc_grades/`), `tfbd_results.json`, accuracy summary.

## Demo benchmark technique

When ARC data is off, `tfbd_generate()` runs TFBD-orchestrated block diffusion:

1. Fiber primitive resonance picks **K=4** trajectories.
2. Fiber-bundle transition smoothing on the torus discourse buffer.
3. K parallel conditioned `generate()` calls with per-slot lens prompts.
4. Verification + cumprod acceptance commits tokens into the canvas.
5. Doppler-guided reallocation on weak slots.

## The 12 spatial primitives

`Rotate90`, `Rotate180`, `ReflectH`, `ReflectV`, `Transpose`, `CropBBox`, `TileRepeat`, `ColorMap`, `SymmetryComplete`, `FloodFill`, `ComponentExtract`, `GravityShift` — each has a prompt lens and fiber-space fingerprint in `FiberPrimitiveAllocator`.

## Key files

| Path | Purpose |
|------|---------|
| `tfbd.py` | Main entry (load, verify, ARC eval, benchmark, CLI) |
| `agent-tools/verify_arc_phase1.py` | CPU-only structure / budget tests |
| `agent-tools/test_pixel_vote.py` | Pixel vote unit tests |

Legacy alias: `million_brains_dflash_generate = tfbd_generate` in `tfbd.py`.

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

The wheel cell uses `--no-deps`; `accelerate` and `safetensors` are required for sharded HF load.

### 3. Run

```python
!python tfbd.py --arc-profile auto --arc-split evaluation
```

Expected banner: `TOPOLOGICAL-FIBER-BUNDLE-DIFFUSION INITIALIZED`.

Benchmark only: `!python tfbd.py --demo-only`

### 4. Hardware

- **26B MoE** uses manual per-layer `device_map` (avoids accelerate tie-weight crashes on DiffusionGemma).
- **4×22GB** works with sequential/batched Phase 1; **A100 80GB** is more comfortable.
- `TFBD_KEEP_ON_CPU = True` keeps fiber modules on CPU when GPUs are full.

## Configuration

All toggles live in the `TOGGLES` block at the top of `tfbd.py`:

- `ENABLE_TFBD` — fiber injection + cohomological stitch vs legacy pixel vote
- `ARC_DATA_PROFILE` — `"auto"` | `"kaggle"` | `"local"` | `"off"`
- `ARC_HYPOTHESIS_SLOTS` — Phase-1 pool size (default 8)
- `ARC_SPATIAL_GRID_ENSEMBLE` — grid hypotheses + Phase-2 fusion
- `EVAL_MAX_TASKS` / `EVAL_SMOKE_TASK_ID` — smoke-test scope
- `K` — demo benchmark parallel trajectories only

## Reading the logs

| Prefix | Meaning |
|--------|---------|
| `[TFBD-DIFFUSION]` | Benchmark denoise loop |
| `[TFBD-FIBER]` | Fiber primitive resonance |
| `[ARC-PHASE-1]` | Spatial hypothesis generation |
| `[ARC-PHASE-2]` | Cohomological stitch or pixel vote |
| `[FINAL][arc]` | Dataset accuracy |

## Verification

```bash
python agent-tools/verify_arc_phase1.py
python agent-tools/test_pixel_vote.py
```

## License / usage

Educational and research prototype. Pull requests welcome.