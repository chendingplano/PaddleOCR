# Phase I: PaddleOCR Timing Instrumentation — Design & Analysis

**Date:** 2026-03-28 \
**File modified:** `pdf_parser_service.py` \
**Status:** Approved

---

## 1. Problem Statement

Parsing a single PDF page with PaddleOCR-VL takes 20–100+ seconds:

```
Start parsing page  1/11, sec: 28
Start parsing page  2/11, sec: 23
Start parsing page  3/11, sec: 53
Start parsing page  4/11, sec: 85
Start parsing page  5/11, sec: 62
Start parsing page  6/11, sec:108
...
```

The wide variance (23s–108s) suggests the bottleneck is proportional to some
per-page property. Code analysis identified it as the number of detected layout
blocks.

---

## 2. Call Chain Analysis

### 2.1 Entry Point

```
PaddleOCRVL().predict(input=img_array)
  File: paddleocr/_pipelines/paddleocr_vl.py:155
  └─ returns list(predict_iter(...))

PaddleOCRVL.predict_iter()
  File: paddleocr/_pipelines/paddleocr_vl.py:98
  └─ self.paddlex_pipeline.predict(input, ...)
      [self.paddlex_pipeline is _PaddleOCRVLPipeline, created in base.py:67]
```

### 2.2 Pipeline Initialization

```
PaddleXPipelineWrapper.__init__()  [base.py:55–109]
  Line 66: self._merged_paddlex_config = self._get_merged_paddlex_config()
           Loads YAML: paddlex/configs/pipelines/PaddleOCR-VL-1.5.yaml
             batch_size: 64
             use_queues: true
             use_layout_detection: true
  Line 67: self.paddlex_pipeline = self._create_paddlex_pipeline()
           Returns: _PaddleOCRVLPipeline instance
```

### 2.3 Complete Call Tree — What Happens When `predict(img_array)` Is Called for One Page

```
_PaddleOCRVLPipeline.predict()     [pipeline.py:511–900]
│
├─ SETUP  [lines 575–600]
│   └─ get_model_settings()         resolve feature flags
│   └─ check_model_settings_valid() validate initialized models
│
├─ SEQUENTIAL MODE (use_queues=False) OR
│  PARALLEL MODE  (use_queues=True, default)
│
│  ┌─ _process_cv(batch_data)       [lines 615–686]   ← CV STAGE
│  │
│  │  Line 625: self.img_reader(instances)
│  │            Read numpy array → list of (H,W,3) arrays
│  │
│  │  Lines 627–638: IF use_doc_preprocessor:
│  │    self.doc_preprocessor_pipeline(image_arrays, ...)
│  │    SubPipeline: _DocPreprocessorPipeline  [doc_preprocessor/pipeline.py:133]
│  │    │
│  │    ├─ IF use_doc_orientation_classify:   [line 162]
│  │    │   Model: PP-LCNet_x1_0_doc_ori
│  │    │   Input:  image array
│  │    │   Output: angle ∈ {0, 90, 180, 270}
│  │    │   Action: rot_img = rotate_image(img, angle)
│  │    │
│  │    └─ IF use_doc_unwarping:              [line 177]
│  │        Model: UVDoc
│  │        Input:  rotated image
│  │        Output: perspective-corrected image
│  │
│  │  Lines 643–654: IF use_layout_detection:
│  │    layout_det_results = list(self.layout_det_model(doc_preprocessor_images, ...))
│  │    Model: PP-DocLayoutV3 (or PP-DocLayoutV2)
│  │    Input:  preprocessed image (H,W,3)
│  │    Output: { "boxes": [ { "cls_id": 0–24,
│  │                           "label": "text|table|chart|formula|...",
│  │                           "score": 0.0–1.0,
│  │                           "coordinate": [x1,y1,x2,y2] } ] }
│  │
│  │  Lines 656–661: gather_imgs()
│  │    Collect embedded images from detected regions
│  │
│  └─ YIELD (input_paths, page_indexes, doc_preprocessor_images,
│            doc_preprocessor_results, layout_det_results, imgs_in_doc)
│
│  ┌─ _process_vlm(results_cv)      [lines 688–722]   ← VLM STAGE (BOTTLENECK)
│  │
│  └─ self.get_layout_parsing_results(...)   [line 704, impl at line 251]
│      │
│      ├─ Lines 281–291: FOR each image + layout_det_result:
│      │   filter_overlap_boxes()
│      │   self.crop_by_boxes()     crop image by each bounding box
│      │   merge_blocks()           optionally merge adjacent blocks
│      │
│      ├─ Lines 293–365: FOR each block:
│      │   Build text query based on label:
│      │     "OCR:"                 text blocks
│      │     "Table Recognition:"   tables
│      │     "Chart Recognition:"   charts
│      │     "Formula Recognition:" formulas
│      │     "Seal Recognition:"    seals
│      │     "Spotting:"            text spotting
│      │   Group blocks into batch_dict_by_pixel[(min_pixels, max_pixels)]
│      │
│      ├─ Lines 374–398: FOR each pixel_key group:  ← THE MAIN LOOP
│      │   batch_results = list(
│      │       self.vl_rec_model.predict(
│      │           [{"image": img, "query": query}, ...],
│      │           use_cache=True, min_pixels=..., max_pixels=...,
│      │           temperature=..., top_p=..., repetition_penalty=...,
│      │           max_new_tokens=...,
│      │       )
│      │   )
│      │   Model: PaddleOCR-VL-0.9B (or PaddleOCR-VL-1.5-0.9B)
│      │   │
│      │   └─ DocVLMPredictor.process()  [doc_vlm/predictor.py:181–276]
│      │       ├─ preprocess: resize/pad to pixel constraints, normalize
│      │       ├─ vision encode: transformer → visual tokens
│      │       ├─ text tokenize: query → token IDs
│      │       ├─ LLM generate: autoregressive decode (use_cache=True)
│      │       └─ postprocess: token IDs → text string
│      │
│      └─ Lines 404–461: FOR each block result:
│          truncate_repetitive_content()    [all blocks]
│          convert_otsl_to_html()           [tables]
│          LaTeX string replacements        [formulas]
│          post_process_for_spotting()      [spotting]
│          → PaddleOCRVLBlock(label, bbox, content)
│
└─ PaddleOCRVLResult({
       "parsing_res_list": [...blocks...],
       "layout_det_res":   {...},
       "doc_preprocessor_res": {...},
       ...
   })                                       [line 747–761]
```

### 2.4 Parallel Mode (use_queues=True, default)

```
Thread 1: _worker_input()   → queue_input
Thread 2: _worker_cv()      queue_input → _process_cv() → queue_cv
Thread 3: _worker_vlm()     queue_cv → accumulate until MAX_NUM_BOXES
                            → _process_vlm() → queue_vlm
Main thread: yield from queue_vlm
```

`MAX_NUM_BOXES` = `vl_rec_model.batch_sampler.batch_size`

---

## 3. Root Cause: Forced batch_size=1 for Local VLM

```python
# paddlex/inference/models/doc_vlm/predictor.py:68–75
if (self.model_name in self.model_group["PaddleOCR-VL"]
        and self.batch_sampler.batch_size > 1):
    logging.warning(
        f"Currently, the {repr(self.model_name)} local model "
        f"only supports batch size of 1. The batch size will be updated to 1."
    )
    self.batch_sampler.batch_size = 1
```

**Effect:** Even though `get_layout_parsing_results` passes all blocks for a
page as one list to `vl_rec_model.predict([...])`, internally the generator
processes **one block at a time** (batch_size=1). The outer `list()` call at
line 384 exhausts the generator serially.

**Consequence:** Total page time ≈ N_blocks × time_per_VLM_forward_pass.

| N blocks | @5s/block | @10s/block |
|----------|-----------|------------|
| 5        | 25s       | 50s        |
| 10       | 50s       | 100s       |
| 20       | 100s      | 200s       |

This exactly matches the observed log variance.

---

## 4. Timing Insertion Points

| # | Stage | Location | What to measure |
|---|-------|----------|-----------------|
| 1 | Image read | `pipeline.py:625` | ms (expected: <5ms) |
| 2 | Doc preprocessor | `pipeline.py:628–638` | ms (often disabled) |
| 3 | Layout detection | `pipeline.py:645` | ms; box count and labels |
| 4 | Block crop + merge | `pipeline.py:284–291` | ms (expected: <10ms) |
| 5 | **VLM inference** | `pipeline.py:384–396` | **ms total; N blocks; avg ms/block** |
| 6 | Post-processing | `pipeline.py:404–461` | ms (expected: <50ms) |

Steps 3 and 5 are the only ones expected to be non-trivial. Steps 1, 4, 6
should collectively be under 100ms for a typical page.

---

## 5. Approach Considered

Three options were evaluated:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A** | Time only `predict()` wall-clock in `pdf_parser_service.py` | Zero risk | Can't distinguish layout vs VLM time |
| **B ✓** | Monkey-patch pipeline object at startup when `PDF_TIMING=true` | All code in service file; full stage breakdown; no installed-file edits | Moderate complexity |
| **C** | Edit `.venv/…/paddlex/…/pipeline.py` directly | Most granular | Modifies pip-installed files; lost on venv recreate |

**Selected: Approach B.**

---

## 6. Design

### 6.1 Flag

```
PDF_TIMING=true    (env var, read via existing _env() helper)
```

Zero overhead when unset — the hooks are never installed.

### 6.2 Entry Point

Single function `_install_timing_hooks(ocr_engine)` called once after engine
initialization in `run()`, inside the `if PDF_TIMING` guard:

```python
if _env("PDF_TIMING", "false").lower() == "true":
    _install_timing_hooks(ocr_engine)
    log.info("timing hooks installed (PDF_TIMING=true)")
```

### 6.3 Object Path

```
ocr_engine                          PaddleOCRVL instance
└─ ocr_engine.paddlex_pipeline      _PaddleOCRVLPipeline instance (base.py:67)
    ├─ .doc_preprocessor_pipeline   _DocPreprocessorPipeline
    ├─ .layout_det_model            PP-DocLayoutV3 model
    ├─ .vl_rec_model                PaddleOCR-VL-0.9B model
    └─ .get_layout_parsing_results  bound method
```

### 6.4 `_TimedModelWrapper`

Used for `doc_preprocessor_pipeline` and `layout_det_model`. Direct instance
patching of `__call__` does not work in Python (special-method lookup goes
through the type, not the instance), so we replace the attribute with a wrapper
object.

```python
class _TimedModelWrapper:
    def __init__(self, model, label: str):
        self._model = model
        self._label = label

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        results = list(self._model(*args, **kwargs))   # exhaust lazy generator
        ms = (time.perf_counter() - t0) * 1000
        if results and isinstance(results[0], dict) and "boxes" in results[0]:
            n = sum(len(r["boxes"]) for r in results)
            log.info("[TIMING] %s: %.0fms, boxes=%d", self._label, ms, n)
        else:
            log.info("[TIMING] %s: %.0fms", self._label, ms)
        return iter(results)           # caller expects an iterable

    def __getattr__(self, name):       # delegate attribute access to real model
        return getattr(self._model, name)
```

### 6.5 `get_layout_parsing_results` Patch

`get_layout_parsing_results` is a regular bound method (not a special method),
so it can be replaced on the instance directly. The wrapper receives
`layout_det_results` as a keyword argument and can count blocks before calling
the original.

```python
_orig = pipeline.get_layout_parsing_results

def _timed_glpr(**kwargs):
    det_results = kwargs.get("layout_det_results", [])
    from collections import Counter
    label_counts = Counter(
        b.get("label", "?")
        for r in det_results
        for b in r.get("boxes", [])
    )
    n_blocks = sum(label_counts.values())
    log.info("[TIMING] Blocks: %s", dict(label_counts))
    t0 = time.perf_counter()
    result = _orig(**kwargs)
    ms = (time.perf_counter() - t0) * 1000
    log.info(
        "[TIMING] VLM: %d blocks, %.0fms total, %.0fms avg/block",
        n_blocks, ms, ms / max(n_blocks, 1),
    )
    return result

pipeline.get_layout_parsing_results = _timed_glpr
```

### 6.6 `_install_timing_hooks` Assembly

```python
def _install_timing_hooks(ocr_engine) -> None:
    pipeline = ocr_engine.paddlex_pipeline

    if hasattr(pipeline, "doc_preprocessor_pipeline"):
        pipeline.doc_preprocessor_pipeline = _TimedModelWrapper(
            pipeline.doc_preprocessor_pipeline, "DocPreproc"
        )

    if hasattr(pipeline, "layout_det_model"):
        pipeline.layout_det_model = _TimedModelWrapper(
            pipeline.layout_det_model, "LayoutDet"
        )

    if hasattr(pipeline, "get_layout_parsing_results"):
        _orig = pipeline.get_layout_parsing_results
        def _timed_glpr(**kwargs):
            ...  # as above
        pipeline.get_layout_parsing_results = _timed_glpr
```

`hasattr` guards ensure the function is silent if PaddleOCR's internal
structure changes.

### 6.7 Expected Log Output (per page)

```
[TIMING] DocPreproc:   12ms
[TIMING] LayoutDet:   430ms, boxes=14
[TIMING] Blocks: {'text': 9, 'table': 1, 'formula': 3, 'title': 1}
[TIMING] VLM:    14 blocks, 72140ms total, 5153ms avg/block
```

---

## 7. Out of Scope (This Round)

- Per-block VLM timing (average ms/block is sufficient; per-block adds complexity)
- Structured output / CSV export
- Timing the fitz page-render step (known-fast)
- Any optimization — this round is data collection only

---

## 8. Files Changed

- `pdf_parser_service.py` — only file modified
  - Add `_TimedModelWrapper` class (~20 lines)
  - Add `_install_timing_hooks()` function (~30 lines)
  - Add 3-line call site in `run()` after engine init

---

## Appendix: Source Files Referenced

| File | Purpose |
|------|---------|
| `paddleocr/_pipelines/paddleocr_vl.py:155` | PaddleOCRVL.predict() entry |
| `paddleocr/_pipelines/paddleocr_vl.py:98` | predict_iter() → paddlex_pipeline |
| `paddleocr/_pipelines/base.py:55–109` | PaddleXPipelineWrapper init |
| `.venv/…/paddlex/inference/pipelines/paddleocr_vl/pipeline.py:511` | _PaddleOCRVLPipeline.predict() |
| `.venv/…/paddlex/inference/pipelines/paddleocr_vl/pipeline.py:615` | _process_cv() |
| `.venv/…/paddlex/inference/pipelines/paddleocr_vl/pipeline.py:251` | get_layout_parsing_results() |
| `.venv/…/paddlex/inference/pipelines/paddleocr_vl/pipeline.py:384` | vl_rec_model.predict() call |
| `.venv/…/paddlex/inference/models/doc_vlm/predictor.py:68–75` | batch_size=1 override |
| `.venv/…/paddlex/inference/pipelines/doc_preprocessor/pipeline.py:133` | DocPreprocessor pipeline |
| `/tmp/paddleocrvl_call_chain.txt` | Initial call chain trace |
| `/tmp/call_tree.txt` | Complete call tree |
| `/tmp/comprehensive_trace.txt` | Comprehensive trace with line refs |
