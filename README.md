# one-million-brains-diffusiongemma

<img width="784" height="1168" alt="image" src="https://github.com/user-attachments/assets/41dd8f5d-7fb1-4437-acd5-ddc00146e6d9" />

**Permutation-Gated Feature-Slot Diffusion** on DiffusionGemma — HuggingFace block-diffusion with a Million-Brains orchestration layer (prompt + sampling conditioning, not draft-model speculative decoding).

## What it is

Single Kaggle script (`million_brains_dflash.py`) that runs **two separate pipelines** on one DiffusionGemma 26B engine:

| Pipeline | Purpose | Technique |
|----------|---------|-----------|
| **ARC eval** (default when data is mounted) | ARC-AGI-2 scoring | Two-phase **spatial grid ensemble** — no Million-Brains denoise loop |
| **Demo benchmark** (`--demo-only` or no ARC data) | Smoke / throughput demo | **K=4** parallel conditioned trajectories per denoise super-block |

- **Engine:** DiffusionGemma 26B via HuggingFace `transformers>=5.12.1` (`DiffusionGemmaForBlockDiffusion`, `INFERENCE_BACKEND = "hf"`). Not vLLM, not causal LM.
- **Primitive bank:** 12 spatial transformation lenses (`Rotate90`, `ReflectH`, `ColorMap`, …) permuted by `PermutationFeatureSlotAllocator`.
- **Conditioning:** Per-slot chat prompts + greedy sampling params. No hidden-state / kernel injection.

## ARC eval technique (default path)

When competition data is attached, the script evaluates ARC-AGI without the benchmark denoise loop:

**Phase 1 — spatial hypothesis pool** (`ARC_HYPOTHESIS_SLOTS = 8`)

1. Hash pooled state → permutation of 8 primitives from the 12-lens bank.
2. For each slot: build a spatial-lens prompt (train pairs + test input + primitive instruction).
3. Sequential `generate()` calls (`ARC_PHASE1_PROMPT_PARALLELISM = False`, one slot per call).
4. Greedy JSON decode (`ARC_GENERATION_TEMPERATURE = 0.0`), `[[` prefill, thinking off.
5. Parse each response into a 2D integer grid.

**Phase 2 — pixel majority vote** (`ARC_SPATIAL_GRID_ENSEMBLE = True`)

- Per-cell plurality across parsed Phase-1 grids → final output grid.
- **No LLM synthesis** in this mode; Phase 2 is deterministic voting only.

Outputs: per-test grade cards (PNG), answer report JSON, accuracy summary.

> `BENCHMARK_K = 4` is **not** used in ARC Phase 1/2. ARC hypothesis count is `ARC_HYPOTHESIS_SLOTS`.

## Demo benchmark technique (Million-Brains denoise)

Only when running `--demo-only` or when ARC paths are unavailable:

At each denoise super-block (`DIFFUSION_DENOISE_CHUNK = 6` tokens):

1. Hash history → permutation of **K=4** features from 12 primitives.
2. **CTSB** circuit smoothing limits slot swaps between super-blocks.
3. **K** parallel `engine.generate()` calls with per-slot lens prompts.
4. Cross-stream fusion + cumprod logprob verification + adaptive reallocation.
5. Accepted tokens commit into the 256-token block-diffusion canvas; loop continues.

## Key files

- `million_brains_dflash.py` — full script (load, verify, ARC eval, optional demo benchmark)
- `agent-tools/` — auxiliary verification scripts (not wired into the main ARC path)

## Quick start (Kaggle)

### 1. Notebook inputs

| Input | Handle |
|-------|--------|
| Model | `google/diffusiongemma` → `diffusiongemma-26b-a4b-it` |
| Competition (eval) | `arc-prize-2026-arc-agi-2` |

### 2. Dependencies (run before the script)

The `godelcomplete/vllm-gemma` wheel bundle installs `transformers==5.12.1` with `--no-deps`. Also install:

```python
!pip install -q "transformers==5.12.1"  # or the vllm-gemma wheel cell
!pip install -q "accelerate>=0.26.0" "safetensors>=0.4.0" "bitsandbytes>=0.43.0"
```

Restart the kernel after installing so `diffusion_gemma` is visible to `transformers`.

### 3. Run

Paste or import `million_brains_dflash.py` and execute. Expect:

```
ONE-MILLION-BRAINS-DIFFUSIONGEMMA INITIALIZED
```

With ARC data mounted, eval runs automatically. For benchmark only:

```bash
python million_brains_dflash.py --demo-only
```

### 4. Hardware notes

- **26B MoE** needs multi-GPU sharding or 4-bit quantization.
- Loader tries, in order: **bitsandbytes 4-bit NF4** → accelerate `from_pretrained` + offload → `load_checkpoint_and_dispatch` → lazy load.
- Writable offload dir: `/kaggle/working/diffusiongemma_offload` (Kaggle inputs are read-only).
- Tested target: 4×22GB GPUs; single A100 80GB also works. `torch._dynamo` is disabled to avoid meta-tensor errors during block-diffusion `generate()`.

## Configuration (top of script)

All toggles live in the `TOGGLES` block at the top of `million_brains_dflash.py`:

- `ARC_DATA_PROFILE` — `"auto"` detects Kaggle mount; `"off"` skips eval
- `ARC_HYPOTHESIS_SLOTS` — Phase-1 pool size (default 8)
- `ARC_SPATIAL_GRID_ENSEMBLE` — pixel vote vs legacy text-hypothesis + LLM final grid
- `EVAL_MAX_TASKS` / `EVAL_SMOKE_TASK_ID` — limit eval scope for smoke tests
- `K` — demo benchmark parallel trajectories only

## License / usage

Educational and research prototype. Pull requests welcome.