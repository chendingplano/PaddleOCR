# Phase II: PaddleOCR VLM Sub-Step Timing & Parallelization Analysis — Design

**Date:** 2026-03-29 \
**File to modify:** `pdf_parser_service.py` \
**Prerequisite:** `2026-03-28-timing-instrumentation-design.md` (Phase 1, already implemented) \
**Status:** Approved

---

## 1. Problem Statement

Phase 1 instrumentation confirmed that VLM inference is the dominant cost and that
total page time scales linearly with block count:

```
[TIMING] VLM: 17 blocks, 62409ms total, 3671ms avg/block
[TIMING] VLM: 26 blocks, 70367ms total, 2706ms avg/block
```

Two questions remain unanswered:

1. **Where inside each block's processing does the time go?** (preprocess vs. GPU
   inference vs. postprocess — each is actionable differently)
2. **Which block types dominate, and can block processing be parallelized to reduce
   end-to-end latency?**

This phase adds instrumentation to answer both questions, and defines the resource
model for a future parallel implementation.

---

## 2. Full Per-Page Pipeline

The current timing covers only stages 5–7. This phase adds the missing stages.

| # | Stage | Code location | Expected cost |
|---|-------|---------------|---------------|
| 1 | PDF page render | `page.get_pixmap()` in `_run_ocr` | ~20–50ms |
| 2 | PNG save to disk | `pix.save(image_path)` in `_run_ocr` | ~10–30ms |
| 3 | Array construction | `np.frombuffer().reshape()` in `_run_ocr` | <5ms |
| 4 | Doc preprocessor | existing hook | optional, usually off |
| 5 | Layout detection | existing `_TimedModelWrapper` | ~900–1300ms |
| 6 | Block prep | inside `get_layout_parsing_results` lines 281–366 | ~50–150ms |
| 7 | **VLM loop** | `DocVLMPredictor.process()` per block | **bottleneck** |
| 8 | Result assembly | inside `get_layout_parsing_results` lines 400–500 | ~50–100ms |

Stages 1–3 are currently not timed. Stage 6 and 8 are measured indirectly as
`total − VLM_loop` once per-block process timing is added in stage 7.

---

## 3. Sub-Step Breakdown of Stage 7 (Per-Block VLM Inference)

`DocVLMPredictor.process()` has four sequential sub-steps for each block
(batch_size=1 forced for local PaddleOCR-VL models):

| Sub-step | Method | What it does | Expected cost |
|----------|--------|--------------|---------------|
| **pre** | `processor.preprocess()` | Resize/pad block image to pixel constraints; tokenize query string | 5–20ms |
| **xfer** | `_switch_inputs_to_device()` | Copy tensors CPU → MPS (unified memory on M4; near-zero) | <5ms |
| **gen** | `infer.generate()` | VLM forward pass + autoregressive token decode | ~2000–5000ms |
| **post** | `processor.postprocess()` | Decode token IDs → text string | <5ms |

The hypothesis is that `gen` accounts for >95% of per-block time. The
instrumentation will confirm or refute this.

---

## 4. Timing Design

### 4.1 Stage 1–3 Timing in `_run_ocr`

Add `if _use_timing()` guards around the three per-page steps before the OCR call:

```python
if _use_timing():
    _t0 = time.perf_counter()
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
if _use_timing():
    _t1 = time.perf_counter()
    log.info("[TIMING] PageRender: %.0fms", (_t1 - _t0) * 1000)
image_path = os.path.join(work_dir, image_filename)
pix.save(image_path)
if _use_timing():
    _t2 = time.perf_counter()
    log.info("[TIMING] PNGSave: %.0fms", (_t2 - _t1) * 1000)
img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
    pix.height, pix.width, 3
)
if _use_timing():
    log.info("[TIMING] ArrayBuild: %.0fms", (time.perf_counter() - _t2) * 1000)
```

### 4.2 Per-Block Sub-Step Timing — `_timing_state` Accumulator

A module-level `threading.local()` accumulator holds per-block rows for the
duration of one `_timed_glpr` call. Each sub-object wrapper appends to it; the
`_timed_glpr` wrapper reads and resets it.

```python
_timing_state = threading.local()
# _timing_state.blocks: list of dicts, one per block processed this page
# _timing_state.vlm_total_ms: float, sum of all block ms this page
```

Initialized at the top of `_timed_glpr`:
```python
_timing_state.blocks = []
_timing_state.vlm_total_ms = 0.0
```

### 4.3 Wrapping `DocVLMPredictor` Sub-Objects

Object path (confirmed from source):
```
pipeline.vl_rec_model              DocVLMPredictor instance
├─ .processor.preprocess           callable — wrap on instance
├─ ._switch_inputs_to_device       bound method — wrap on instance
├─ .infer.generate                 callable — wrap on infer object
└─ .processor.postprocess          callable — wrap on instance
```

Each wrapper follows the same pattern: record entry time, call original, record
exit time, store delta into a per-call context dict. A `_current_block_timing`
thread-local dict is set up at the start of each `process()` call and read at the
end to emit one log line per block.

Wrapper for `infer.generate` (the most critical sub-step):

```python
_orig_generate = pipeline.vl_rec_model.infer.generate

def _timed_generate(data, **kwargs):
    t0 = time.perf_counter()
    result = _orig_generate(data, **kwargs)
    _timing_state.current_block["gen_ms"] = (time.perf_counter() - t0) * 1000
    return result

pipeline.vl_rec_model.infer.generate = _timed_generate
```

The same pattern applies to `processor.preprocess`, `_switch_inputs_to_device`,
and `processor.postprocess`.

### 4.4 Per-Block Log Line (emitted after each `process()` call)

```
[TIMING] Block label=text    pixels=142080  pre=11ms xfer=1ms gen=2841ms post=1ms chars=183
[TIMING] Block label=table   pixels=318400  pre=18ms xfer=2ms gen=4120ms post=2ms chars=612
[TIMING] Block label=formula pixels= 89600  pre= 8ms xfer=1ms gen=1953ms post=0ms chars= 47
```

Fields:
- `label` — block type from the layout detection result (e.g. `text`, `table`, `header`, `paragraph_title`); taken directly from `block["label"]`, not derived from the query prefix
- `pixels` — H × W of the cropped block image
- `pre/xfer/gen/post` — sub-step ms
- `chars` — output character count (proxy for decode length, which drives `gen`)

### 4.5 Per-Block-Type Aggregation (emitted at end of `_timed_glpr`)

Aggregated from `_timing_state.blocks` after all blocks complete:

```
[TIMING] BlockType text:              14 blocks  avg=2654ms  avg_px=127K  avg_chars=142  total=37156ms
[TIMING] BlockType paragraph_title:   2 blocks  avg=2100ms  avg_px= 89K  avg_chars= 45  total= 4200ms
[TIMING] BlockType table:             1 block   avg=4120ms  avg_px=318K  avg_chars=612  total= 4120ms
[TIMING] BlockType header:            1 block   avg=1800ms  avg_px= 65K  avg_chars= 22  total= 1800ms
[TIMING] BlockType number:            1 block   avg= 980ms  avg_px= 22K  avg_chars=  4  total=  980ms
```

Helper: `_log_block_type_summary(blocks: list[dict]) -> None`

### 4.6 Three-Phase Split (emitted at end of `_timed_glpr`)

```
[TIMING] Phases: vlm_loop=42036ms  other=215ms  total=42251ms
```

- `vlm_loop` = `_timing_state.vlm_total_ms` (sum of all per-block process durations)
- `total` = wall-clock of entire `get_layout_parsing_results` call
- `other` = `total − vlm_loop` (covers both block prep and result assembly; these two
  cannot be separated without modifying the pipeline source file)

### 4.7 Complete Expected Log Output (one page example)

```
[TIMING] PageRender:  38ms
[TIMING] PNGSave:     22ms
[TIMING] ArrayBuild:   2ms
[TIMING] LayoutDet:  967ms, boxes=17
[TIMING] Blocks: {'header': 1, 'paragraph_title': 1, 'text': 14, 'number': 1}
[TIMING] Block label=text          pixels=142080  pre=11ms xfer=1ms gen=2841ms post=1ms chars=183
[TIMING] Block label=text          pixels=118400  pre= 9ms xfer=1ms gen=2210ms post=1ms chars= 94
... (one line per VLM-processed block)
[TIMING] Block label=paragraph_title pixels=89600 pre= 8ms xfer=1ms gen=2100ms post=0ms chars= 45
[TIMING] Block label=header        pixels= 65200  pre= 6ms xfer=0ms gen=1800ms post=0ms chars= 22
[TIMING] Block label=number        pixels= 22400  pre= 4ms xfer=0ms gen= 980ms post=0ms chars=  4
[TIMING] BlockType text:           14 blocks  avg=2654ms  avg_px=127K  avg_chars=142  total=37156ms
[TIMING] BlockType paragraph_title: 1 block   avg=2100ms  avg_px= 89K  avg_chars= 45  total= 2100ms
[TIMING] BlockType header:          1 block   avg=1800ms  avg_px= 65K  avg_chars= 22  total= 1800ms
[TIMING] BlockType number:          1 block   avg= 980ms  avg_px= 22K  avg_chars=  4  total=  980ms
[TIMING] Phases: vlm_loop=42036ms  other=215ms  total=42251ms
[TIMING] VLM: 17 blocks, 42036ms total, 2474ms avg/block
```

---

## 5. Parallelization Analysis

### 5.1 What the Collected Data Will Determine

| Question | Signal to look for |
|----------|--------------------|
| Which block type is the bottleneck? | `BlockType` rows: highest `total` ms |
| Is per-block time driven by image size? | Correlation of `pixels` vs `gen` ms across blocks |
| Is per-block time driven by output length? | Correlation of `chars` vs `gen` ms across blocks |
| Is `gen` truly dominant? | `pre`, `xfer`, `post` should each be <20ms |
| Are `xfer` costs worth optimizing? | If `xfer ≈ 0ms` on M4 unified memory, confirmed free |

### 5.2 Block Independence (Structural Analysis)

All VLM blocks within a page are **structurally independent**:
- Each block is a spatially distinct, cropped image region
- Each VLM call has no dependency on the results of other blocks on the same page
- The `DocVLMPredictor` model is stateless across calls (KV cache is not shared
  between blocks)

This means intra-page block parallelism is architecturally safe. The only
constraint is hardware concurrency (MPS scheduler, memory capacity).

### 5.3 Parallelization Opportunities (Priority Order)

**P1 — Intra-page VLM block parallelism**
Multiple VLM workers process different blocks of the same page concurrently.
Each worker runs one `DocVLMPredictor.process()` call. Workers are independent;
output is collected and merged in original block order.

Expected speedup: if N workers and `gen` dominates, page time ≈ `ceil(B/N) ×
avg_gen`. For 17 blocks at avg 2474ms with 3 workers: 6 rounds × 2474ms ≈ 14.8s
vs. 42s serial.

**P2 — Inter-page pipelining (already partially present)**
The existing `_worker_cv` / `_worker_vlm` queue threads overlap layout detection
for page N+1 with VLM processing for page N. This is already in place but the
queue is bounded by `MAX_NUM_BOXES`. No new implementation needed for P2; it is
improved automatically by P1 reducing VLM time.

**P3 — PDF render + PNG save offload**
`get_pixmap()` and `pix.save()` are CPU-bound and can run on a separate thread
while VLM processes the prior page. Currently in `_run_ocr` they block the main
loop. Only worth implementing after P1 if stage-1/2 timing shows >100ms cost.

### 5.4 Block-Type Routing (Future Optimization)

If per-block-type data shows that `header`, `footer`, `number`, and
`paragraph_title` blocks have short output (< ~50 chars) and small images
(< ~100K pixels), a future optimization could route these to a lightweight
traditional OCR engine (e.g. Tesseract or easyocr) instead of the full VLM,
reserving VLM workers for `table`, `formula`, and `text` blocks. This is out of
scope for this phase but the `avg_chars` and `avg_px` data collected here will
quantify the opportunity.

---

## 6. Resource Model — M4, 48 GB Unified Memory

### 6.1 Per-VLM-Worker Memory Estimate

| Component | Size |
|-----------|------|
| Model weights (bfloat16, 0.9B params) | ~1.8 GB |
| KV cache + activation buffers during decode | ~1.5–2.5 GB |
| **Total per active VLM worker** | **~3.5–4.5 GB** |

### 6.2 Shared Resources (not replicated per worker)

| Component | Size |
|-----------|------|
| Layout detection model (PP-DocLayoutV3) | ~200 MB |
| Per-page image buffer at 2× scale (letter) | ~40–80 MB |
| OS + Python runtime | ~2–3 GB |

### 6.3 Worker Count Budget

```
Available:  48 GB × 0.75 safety margin = 36 GB
Shared:      3 GB (OS + layout det + page buffer)
Per-worker:  4 GB
Maximum workers:  floor((36 − 3) / 4) = 8
Practical target: 3–4  (leaves headroom for KV cache growth and other processes)
```

### 6.4 Configurable Worker Count

The number of parallel VLM workers is controlled by a new env var:

```
PDF_VLM_WORKERS=<int>    Number of concurrent VLM worker threads.
                         Default: 1 (serial, current behavior).
                         Recommended range: 2–4 on M4/48 GB.
                         Set to 1 to disable parallelism entirely.
```

Read via the existing `_env()` helper:

```python
def _vlm_workers() -> int:
    try:
        n = int(_env("PDF_VLM_WORKERS", "1"))
        return max(1, n)
    except ValueError:
        return 1
```

This value is read at startup and passed to the parallel executor. Setting it to 1
preserves the current serial behavior with zero overhead. The implementation plan
for parallel execution (using `concurrent.futures.ThreadPoolExecutor` or a
`queue.Queue`-based worker pool) is deferred to the next phase once timing data
confirms the expected speedup.

---

## 7. Out of Scope (This Round)

- Actual parallel VLM implementation — this round is instrumentation + analysis only
- Block-type routing to alternative OCR engines
- Structured output / CSV export of timing data
- Multi-GPU or multi-process parallelism
- Tuning `max_new_tokens` per block type

---

## 8. Files Changed

- `pdf_parser_service.py` — only file modified
  - Add `import threading` (if not already imported)
  - Add `_timing_state = threading.local()` module-level
  - Add `_vlm_workers()` config helper
  - Add `_log_block_type_summary()` helper (~15 lines)
  - Extend `_install_timing_hooks()`:
    - Wrap `vl_rec_model.processor.preprocess`
    - Wrap `vl_rec_model._switch_inputs_to_device`
    - Wrap `vl_rec_model.infer.generate`
    - Wrap `vl_rec_model.processor.postprocess`
    - Update `_timed_glpr` to initialize/read `_timing_state`; emit per-block-type summary and 3-phase split
  - Extend `_run_ocr()`: add stage 1–3 timing guards

---

## 9. Appendix: Block Label → Query Prefix Mapping

Used to derive `label` in per-block log lines from the query prompt string:

| Query prefix | Block label in log |
|---|---|
| `OCR:` | `text` (and all other non-special labels: `paragraph_title`, `header`, `footer`, `doc_title`, `number`, `content`, etc.) |
| `Table Recognition:` | `table` |
| `Chart Recognition:` | `chart` |
| `Formula Recognition:` | `formula` |
| `Spotting:` | `spotting` |
| `Seal Recognition:` | `seal` |

Note: the actual block label (from layout detection) is more specific than the
query prefix. The per-block log line should use the original `block_label` from
the block dict, not the derived query prefix, for maximum diagnostic value.

---

## 10. Source Files Referenced

| File | Purpose |
|------|---------|
| `pdf_parser_service.py:391–426` | `_run_ocr` — stages 1–3 and OCR call |
| `pdf_parser_service.py:268–339` | `_install_timing_hooks` — existing Phase 1 hooks |
| `.venv/…/paddleocr_vl/pipeline.py:251–500` | `get_layout_parsing_results` — stages 6–8 |
| `.venv/…/doc_vlm/predictor.py:181–276` | `DocVLMPredictor.process` — stage 7 sub-steps |
| `.venv/…/doc_vlm/predictor.py:66` | `self.infer, self.processor = self._build(...)` |
