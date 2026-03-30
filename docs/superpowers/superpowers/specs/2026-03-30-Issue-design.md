# Phase IV: torchao Weight-Only Quantization — Design

**Date:** 2026-03-30 \
**File to modify:** `pdf_parser_service.py` \
**Prerequisites:**
- `2026-03-29-mps-acceleration-and-dynamic-pixels-design.md` (Phase III, implemented)
- `2026-03-30-Issue.md` (root cause investigation that motivated this phase)

---

## 1. Root Cause Summary (from Issue Investigation)

Phase III delivered only ~1.8× improvement against a ~10× target. Investigation showed two root causes:

**Root cause 1 — The "8–12× MPS speedup" estimate was based on CUDA assumptions.**
On Apple M4, CPU and GPU share the same unified memory bus (~68 GB/s). At `batch_size=1`
(the only mode PaddleOCR-VL supports), transformer inference is memory-bandwidth bound: each
forward pass reads all 3.6 GB of FP32 weights from DRAM through that shared bus. MPS and CPU
compete for the same pipe, so moving to MPS provides no bandwidth advantage.

**Root cause 2 — Dynamic pixels (Phase III) provides the full 1.8× gain.**
Reducing prefill tokens from 576 → 256 (2.25×) lowers total gen time from ~2993 ms to
~1654 ms avg/block. The ~1.8× matches CPU-only math for fewer input tokens exactly; MPS
contributes nothing measurable on top.

**What actually limits speed:** memory bandwidth. Each decode step loads 3.6 GB of FP32
weights through a 68 GB/s bus (~53 ms/decode-step). Quantization directly attacks this.

---

## 2. Solution Overview

Add `_enable_quantization(ocr_engine)` to `pdf_parser_service.py` as a monkey-patch,
following the identical pattern as `_enable_mps_acceleration`.

Apply `torchao.quantization.quantize_(vl_model.infer, config)` once at startup.
The model stays quantized in memory for the lifetime of the process. Comparing quantized
vs. unquantized is done by running the service twice with different env vars.

| Env var | Default | Effect |
|---|---|---|
| `PDF_QUANTIZE_ENABLED` | `false` | Set `true` to enable quantization |
| `PDF_QUANTIZE_BITS` | `8` | `8` = INT8 weight-only; `4` = INT4 weight-only |

**Call order** in `run()`, after engine init:

```python
if _use_quantization():
    _enable_quantization(ocr_engine)   # CPU weights quantized first
if _use_mps():
    _enable_mps_acceleration(ocr_engine)  # then optionally move to MPS
_install_dynamic_pixels_hook(ocr_engine)  # always
if _use_timing():
    _install_timing_hooks(ocr_engine)  # last, so it measures after all opts
```

Quantization must precede MPS because torchao's `quantize_()` targets CPU tensors; moving
the quantized model to MPS afterward is safe, but quantizing after `.to("mps")` risks
unsupported operations in torchao.

---

## 3. Expected Impact

| Scheme | Weight size | Bytes per fwd pass | Decode step (est.) | Speedup vs FP32 |
|---|---|---|---|---|
| FP32 (current) | 3.6 GB | 3.6 GB | ~53 ms/token | 1× |
| INT8 weight-only | 1.8 GB | 1.8 GB | ~27 ms/token | ~2× |
| INT4 weight-only | 0.9 GB | 0.9 GB | ~13 ms/token | ~4× (see note) |

**Note on INT4:** torchao's INT4 GEMM kernels (`_int4_mm`) are CUDA-optimized. On Apple
Silicon CPU the fallback path may not outperform INT8. Timing hooks will show the real
number immediately; `PDF_QUANTIZE_BITS=4` is provided but results should be validated
before relying on it.

Combined with the existing DynPx 1.8× gain, INT8 quantization targets ~3–4× total
improvement over the original baseline.

---

## 4. Design: Config Helpers

```python
def _use_quantization() -> bool:
    return _env("PDF_QUANTIZE_ENABLED", "false").lower() == "true"


def _quantize_bits() -> int:
    """Return 4 or 8. Any value other than '4' is treated as 8."""
    return 4 if _env("PDF_QUANTIZE_BITS", "8").strip() == "4" else 8
```

---

## 5. Design: `_enable_quantization`

```python
def _enable_quantization(ocr_engine) -> None:
    """Quantize VLM model weights using torchao weight-only quantization.

    Must be called BEFORE _enable_mps_acceleration: torchao quantize_() operates
    on CPU tensors; moving the quantized model to MPS afterward is safe.

    PDF_QUANTIZE_BITS=8  → Int8WeightOnlyConfig  (recommended; 2× memory savings,
                           reliable on Apple Silicon CPU via NEON)
    PDF_QUANTIZE_BITS=4  → Int4WeightOnlyConfig  (4× memory savings; torchao INT4
                           GEMM is CUDA-optimized — verify timing on CPU before
                           relying on this)

    On any failure (import error, quantize_() raises), logs a warning and leaves
    the model unchanged — no crash.
    """
    try:
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
    except ImportError:
        log.warning("[Quant] torchao not installed — quantization skipped")
        return

    outer = ocr_engine.paddlex_pipeline
    pipeline = getattr(outer, "_pipeline", outer)

    if not hasattr(pipeline, "vl_rec_model"):
        log.warning("[Quant] vl_rec_model not found — skipping")
        return

    vl_model = pipeline.vl_rec_model

    if getattr(vl_model, "_quantization_enabled", False):
        log.warning("[Quant] already enabled — skipping")
        return

    bits = _quantize_bits()
    config = Int8WeightOnlyConfig() if bits == 8 else Int4WeightOnlyConfig()
    label = f"INT{bits}"

    try:
        quantize_(vl_model.infer, config)
    except Exception as exc:
        log.warning("[Quant] quantize_() failed (%s): %s — model stays FP32", label, exc)
        return

    vl_model._quantization_enabled = True
    log.info("[Quant] model quantized to %s weight-only", label)
```

---

## 6. Failure Modes

| Failure | Behavior |
|---|---|
| `torchao` not installed | Warning logged; model stays FP32 |
| `vl_rec_model` absent (API change) | Warning logged; silent skip |
| `quantize_()` raises (unsupported op or config) | Warning logged; model stays FP32 |
| Double install | Warning logged; second call is no-op |

---

## 7. Testing

New test class `TestEnableQuantization` in `tests/test_timing_hooks.py`:

| Test | What it verifies |
|---|---|
| `test_quantization_int8_applied` | `quantize_()` called with `Int8WeightOnlyConfig` when `PDF_QUANTIZE_BITS=8` |
| `test_quantization_int4_applied` | `quantize_()` called with `Int4WeightOnlyConfig` when `PDF_QUANTIZE_BITS=4` |
| `test_quantization_import_error_is_noop` | `ImportError` from torchao → model unchanged, warning logged |
| `test_quantization_quantize_raises_is_noop` | `quantize_()` exception → model unchanged, warning logged |
| `test_quantization_double_install_skipped` | Second call is no-op; warning logged |
| `test_use_quantization_env_true` | `PDF_QUANTIZE_ENABLED=true` → returns `True` |
| `test_use_quantization_env_false` | Unset or `false` → returns `False` |
| `test_quantize_bits_default` | Unset → returns `8` |
| `test_quantize_bits_4` | `PDF_QUANTIZE_BITS=4` → returns `4` |
| `test_quantize_bits_invalid` | Non-`4` value → returns `8` |

---

## 8. Files Changed

- `pyproject.toml` — add `torchao` to the `service` extra
- `pdf_parser_service.py`
  - Add `_use_quantization()` helper (~3 lines)
  - Add `_quantize_bits()` helper (~3 lines)
  - Add `_enable_quantization()` function (~40 lines)
  - Modify `run()`: add 2-line call site before MPS hook, reorder to quantize-first

---

## 9. Env Var Reference

| Var | Default | Description |
|---|---|---|
| `PDF_QUANTIZE_ENABLED` | `false` | Set `true` to apply torchao weight-only quantization at startup |
| `PDF_QUANTIZE_BITS` | `8` | Quantization precision: `8` = INT8, `4` = INT4 |
| `PDF_MPS` | `false` | Existing: move VLM to Apple Metal GPU |
| `PDF_VLM_OCR_MAX_PIXELS` | `50176` | Existing: cap min_pixels for OCR blocks |
| `PDF_TIMING` | `false` | Existing: enable per-stage timing hooks |
