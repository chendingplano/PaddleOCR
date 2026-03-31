# Timing Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PDF_TIMING=true` instrumentation to `pdf_parser_service.py` that logs wall-clock time for layout detection, block counts, and VLM inference per page without modifying any paddlex installed files.

**Architecture:** A `_TimedModelWrapper` class wraps callable model objects (used for `doc_preprocessor_pipeline` and `layout_det_model`). A separate `_install_timing_hooks()` function replaces the `get_layout_parsing_results` bound method with a closure that counts blocks and times VLM inference. Both are wired into `run()` behind a `PDF_TIMING=true` env var guard.

**Tech Stack:** Python 3.12, uv (package/run manager), pytest for unit tests, `std_513428.pdf` for live verification.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pdf_parser_service.py` | Modify | Add `_TimedModelWrapper`, `_install_timing_hooks()`, call site in `run()`, docstring update |
| `tests/test_timing_hooks.py` | Create | Unit tests for wrapper and hook installation using mock objects |

---

### Task 1: Unit tests for `_TimedModelWrapper`

**Files:**
- Create: `tests/test_timing_hooks.py`

- [ ] **Step 1: Create `tests/test_timing_hooks.py` with failing tests for `_TimedModelWrapper`**

```python
"""Unit tests for timing instrumentation helpers in pdf_parser_service."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _TimedModelWrapper tests
# ---------------------------------------------------------------------------

class TestTimedModelWrapper:
    def _make_wrapper(self, return_value, label="TestModel"):
        from pdf_parser_service import _TimedModelWrapper
        mock_model = MagicMock(return_value=iter(return_value))
        return _TimedModelWrapper(mock_model, label), mock_model

    def test_calls_underlying_model_with_same_args(self):
        wrapper, mock_model = self._make_wrapper([{"boxes": []}])
        list(wrapper("arg1", key="val"))
        mock_model.assert_called_once_with("arg1", key="val")

    def test_returns_iterable_of_original_results(self):
        fake_results = [{"boxes": [{"label": "text"}]}, {"boxes": []}]
        wrapper, _ = self._make_wrapper(fake_results)
        results = list(wrapper())
        assert results == fake_results

    def test_logs_ms_and_box_count_for_layout_det_results(self, caplog):
        fake_results = [
            {"boxes": [{"label": "text"}, {"label": "table"}]},
            {"boxes": [{"label": "formula"}]},
        ]
        wrapper, _ = self._make_wrapper(fake_results, label="LayoutDet")
        with caplog.at_level(logging.INFO):
            list(wrapper())
        assert any("LayoutDet" in r.message and "boxes=3" in r.message
                   for r in caplog.records)

    def test_logs_ms_only_when_results_have_no_boxes_key(self, caplog):
        fake_results = [{"output_img": "array"}]
        wrapper, _ = self._make_wrapper(fake_results, label="DocPreproc")
        with caplog.at_level(logging.INFO):
            list(wrapper())
        # Should log timing but NOT "boxes="
        timing_records = [r for r in caplog.records if "DocPreproc" in r.message]
        assert len(timing_records) == 1
        assert "boxes=" not in timing_records[0].message

    def test_delegates_attribute_access_to_wrapped_model(self):
        from pdf_parser_service import _TimedModelWrapper
        mock_model = MagicMock()
        mock_model.batch_sampler = "sentinel"
        wrapper = _TimedModelWrapper(mock_model, "X")
        assert wrapper.batch_sampler == "sentinel"
```

- [ ] **Step 2: Run tests — expect ImportError (class not defined yet)**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
uv run pytest tests/test_timing_hooks.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name '_TimedModelWrapper' from 'pdf_parser_service'`

---

### Task 2: Implement `_TimedModelWrapper`

**Files:**
- Modify: `pdf_parser_service.py` — add class after the `# OCR` section, before `_count_pdf_pages` (around line 228)

- [ ] **Step 1: Add `_TimedModelWrapper` class to `pdf_parser_service.py`**

Insert after the `# ---------------------------------------------------------------------------` OCR section header comment and before `def _run_ocr`:

```python
class _TimedModelWrapper:
    """Wraps a callable model to log wall-clock time and result count.

    Replaces the model attribute on the pipeline object so that Python's
    normal special-method dispatch (which bypasses instance __call__) is
    avoided.  Attribute access is delegated to the wrapped model so that
    callers accessing e.g. .batch_sampler still work.
    """

    def __init__(self, model, label: str) -> None:
        self._model = model
        self._label = label

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        results = list(self._model(*args, **kwargs))
        ms = (time.perf_counter() - t0) * 1000
        if results and isinstance(results[0], dict) and "boxes" in results[0]:
            n = sum(len(r["boxes"]) for r in results)
            log.info("[TIMING] %s: %.0fms, boxes=%d", self._label, ms, n)
        else:
            log.info("[TIMING] %s: %.0fms", self._label, ms)
        return iter(results)

    def __getattr__(self, name: str):
        return getattr(self._model, name)
```

- [ ] **Step 2: Run tests — expect tests to pass**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
uv run pytest tests/test_timing_hooks.py::TestTimedModelWrapper -v
```

Expected:
```
tests/test_timing_hooks.py::TestTimedModelWrapper::test_calls_underlying_model_with_same_args PASSED
tests/test_timing_hooks.py::TestTimedModelWrapper::test_returns_iterable_of_original_results PASSED
tests/test_timing_hooks.py::TestTimedModelWrapper::test_logs_ms_and_box_count_for_layout_det_results PASSED
tests/test_timing_hooks.py::TestTimedModelWrapper::test_logs_ms_only_when_results_have_no_boxes_key PASSED
tests/test_timing_hooks.py::TestTimedModelWrapper::test_delegates_attribute_access_to_wrapped_model PASSED
5 passed
```

- [ ] **Step 3: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
git add pdf_parser_service.py tests/test_timing_hooks.py
git commit -m "feat: add _TimedModelWrapper for pipeline stage timing"
```

---

### Task 3: Unit tests for `_install_timing_hooks`

**Files:**
- Modify: `tests/test_timing_hooks.py` — append new test class

- [ ] **Step 1: Append `TestInstallTimingHooks` to `tests/test_timing_hooks.py`**

```python
# ---------------------------------------------------------------------------
# _install_timing_hooks tests
# ---------------------------------------------------------------------------

class TestInstallTimingHooks:
    def _make_mock_engine(self, has_preprocessor=True, has_layout=True, has_glpr=True):
        """Build a mock ocr_engine with a fake paddlex_pipeline."""
        pipeline = MagicMock()
        if not has_preprocessor:
            del pipeline.doc_preprocessor_pipeline
        if not has_layout:
            del pipeline.layout_det_model
        if not has_glpr:
            del pipeline.get_layout_parsing_results
        engine = MagicMock()
        engine.paddlex_pipeline = pipeline
        return engine, pipeline

    def test_layout_det_model_is_replaced_with_wrapper(self):
        from pdf_parser_service import _install_timing_hooks, _TimedModelWrapper
        engine, pipeline = self._make_mock_engine()
        original = pipeline.layout_det_model
        _install_timing_hooks(engine)
        assert isinstance(pipeline.layout_det_model, _TimedModelWrapper)
        assert pipeline.layout_det_model._model is original

    def test_doc_preprocessor_is_replaced_with_wrapper(self):
        from pdf_parser_service import _install_timing_hooks, _TimedModelWrapper
        engine, pipeline = self._make_mock_engine()
        original = pipeline.doc_preprocessor_pipeline
        _install_timing_hooks(engine)
        assert isinstance(pipeline.doc_preprocessor_pipeline, _TimedModelWrapper)
        assert pipeline.doc_preprocessor_pipeline._model is original

    def test_get_layout_parsing_results_is_replaced_with_closure(self):
        from pdf_parser_service import _install_timing_hooks, _TimedModelWrapper
        engine, pipeline = self._make_mock_engine()
        original = pipeline.get_layout_parsing_results
        _install_timing_hooks(engine)
        # Should NOT be a _TimedModelWrapper — it's a plain closure
        assert not isinstance(pipeline.get_layout_parsing_results, _TimedModelWrapper)
        assert pipeline.get_layout_parsing_results is not original

    def test_glpr_closure_calls_original_and_returns_result(self, caplog):
        from pdf_parser_service import _install_timing_hooks
        engine, pipeline = self._make_mock_engine()
        sentinel = object()
        pipeline.get_layout_parsing_results.return_value = sentinel
        _install_timing_hooks(engine)
        result = pipeline.get_layout_parsing_results(
            layout_det_results=[{"boxes": [{"label": "text"}]}],
            images=[],
            imgs_in_doc=[],
        )
        assert result is sentinel
        pipeline.get_layout_parsing_results  # call count tracked via mock

    def test_glpr_closure_logs_block_count_and_timing(self, caplog):
        from pdf_parser_service import _install_timing_hooks
        engine, pipeline = self._make_mock_engine()
        pipeline.get_layout_parsing_results.return_value = None
        _install_timing_hooks(engine)
        with caplog.at_level(logging.INFO):
            pipeline.get_layout_parsing_results(
                layout_det_results=[
                    {"boxes": [{"label": "text"}, {"label": "table"}]},
                    {"boxes": [{"label": "formula"}]},
                ],
                images=[],
                imgs_in_doc=[],
            )
        block_log = next((r for r in caplog.records if "Blocks:" in r.message), None)
        vlm_log   = next((r for r in caplog.records if "VLM:" in r.message), None)
        assert block_log is not None
        assert vlm_log is not None
        assert "3" in vlm_log.message   # 3 total blocks

    def test_missing_attributes_are_silently_skipped(self):
        from pdf_parser_service import _install_timing_hooks
        engine, pipeline = self._make_mock_engine(
            has_preprocessor=False, has_layout=False, has_glpr=False
        )
        # Should not raise
        _install_timing_hooks(engine)
```

- [ ] **Step 2: Run new tests — expect ImportError (`_install_timing_hooks` not defined)**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
uv run pytest tests/test_timing_hooks.py::TestInstallTimingHooks -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_install_timing_hooks' from 'pdf_parser_service'`

---

### Task 4: Implement `_install_timing_hooks`

**Files:**
- Modify: `pdf_parser_service.py` — add function after `_TimedModelWrapper` class

- [ ] **Step 1: Add `_install_timing_hooks` function immediately after `_TimedModelWrapper`**

```python
def _install_timing_hooks(ocr_engine) -> None:
    """Monkey-patch key pipeline methods to emit [TIMING] log lines.

    Activated only when PDF_TIMING=true.  All patches are applied to the
    live pipeline *instance* — the class and installed paddlex files are
    not modified.

    Log output per page (example):
        [TIMING] DocPreproc:  12ms
        [TIMING] LayoutDet:  430ms, boxes=14
        [TIMING] Blocks: {'text': 9, 'table': 1, 'formula': 3, 'title': 1}
        [TIMING] VLM:    14 blocks, 72140ms total, 5153ms avg/block
    """
    from collections import Counter

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
        _orig_glpr = pipeline.get_layout_parsing_results

        def _timed_glpr(**kwargs):
            det_results = kwargs.get("layout_det_results", [])
            label_counts = Counter(
                b.get("label", "?")
                for r in det_results
                for b in r.get("boxes", [])
            )
            n_blocks = sum(label_counts.values())
            log.info("[TIMING] Blocks: %s", dict(label_counts))
            t0 = time.perf_counter()
            result = _orig_glpr(**kwargs)
            ms = (time.perf_counter() - t0) * 1000
            log.info(
                "[TIMING] VLM: %d blocks, %.0fms total, %.0fms avg/block",
                n_blocks, ms, ms / max(n_blocks, 1),
            )
            return result

        pipeline.get_layout_parsing_results = _timed_glpr
```

- [ ] **Step 2: Run all timing tests**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
uv run pytest tests/test_timing_hooks.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 3: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
git add pdf_parser_service.py tests/test_timing_hooks.py
git commit -m "feat: add _install_timing_hooks for per-stage VLM timing"
```

---

### Task 5: Wire up call site and update docstring

**Files:**
- Modify: `pdf_parser_service.py:1–23` — add `PDF_TIMING` to the module docstring
- Modify: `pdf_parser_service.py:606–615` — add call site after engine init in `run()`

- [ ] **Step 1: Add `PDF_TIMING` to the module docstring**

In the docstring at the top of the file, add after the `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` entry:

```
  PDF_TIMING           Set to "true" to log per-stage timing (layout detection,
                       block count, VLM inference ms) to help profile slowness.
```

- [ ] **Step 2: Add the call site in `run()` after both engine init branches**

Locate the block in `run()` (around line 606–615):

```python
    if use_vl:
        ocr_engine = PaddleOCRVL()
        log.info("OCR mode initialized: PaddleOCR-VL")
    else:
        ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        log.info("OCR mode initialized: PaddleOCR")
```

Replace with:

```python
    if use_vl:
        ocr_engine = PaddleOCRVL()
        log.info("OCR mode initialized: PaddleOCR-VL")
    else:
        ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        log.info("OCR mode initialized: PaddleOCR")

    if _env("PDF_TIMING", "false").lower() == "true":
        _install_timing_hooks(ocr_engine)
        log.info("timing hooks installed (PDF_TIMING=true)")
```

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
uv run pytest tests/test_timing_hooks.py -v
```

Expected: 11 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
git add pdf_parser_service.py
git commit -m "feat: wire PDF_TIMING=true call site into run() and document env var"
```

---

### Task 6: Live verification with sample PDF

**Files:**
- Read: `std_513428.pdf` (11-page PDF already in the project directory)

- [ ] **Step 1: Run the service in timing mode against the sample DB record**

Set `PDF_TIMING=true` and start the service. In a terminal:

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
PDF_TIMING=true uv run python pdf_parser_service.py 2>&1 | grep -E "\[TIMING\]|Start parsing"
```

- [ ] **Step 2: Verify expected log pattern appears**

For each page you should see a group of lines like:

```
[TIMING] LayoutDet:  NNNms, boxes=N
[TIMING] Blocks: {'text': N, ...}
[TIMING] VLM:    N blocks, NNNNNms total, NNNNms avg/block
Start parsing page X/11, sec:NN (MID_26032733)
```

Confirm:
- `LayoutDet` ms is under 2000ms (should be ~300–600ms)
- `boxes=` count matches the block count in the `Blocks:` line
- `VLM avg/block` × `N blocks` ≈ total page seconds from `Start parsing page` log
- No Python exceptions or tracebacks

- [ ] **Step 3: If `PDF_TIMING` env var is NOT set, confirm no timing lines appear**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
uv run python pdf_parser_service.py 2>&1 | grep "\[TIMING\]"
```

Expected: no output (zero matches).

---

## Self-Review

**Spec coverage:**
- ✓ `PDF_TIMING=true` flag — Task 5
- ✓ `_TimedModelWrapper` for `doc_preprocessor_pipeline` and `layout_det_model` — Tasks 1–2
- ✓ `get_layout_parsing_results` closure for block count + VLM timing — Tasks 3–4
- ✓ `hasattr` guards for missing attributes — Task 4, tested in Task 3
- ✓ Call site in `run()` after engine init — Task 5
- ✓ Module docstring update — Task 5
- ✓ Zero overhead when flag is unset — Task 6 Step 3

**Placeholder scan:** None found — all steps include concrete code.

**Type consistency:**
- `_TimedModelWrapper` used by name in both test imports and `_install_timing_hooks` — consistent.
- `_install_timing_hooks(ocr_engine)` signature matches call site `_install_timing_hooks(ocr_engine)` — consistent.
- `_timed_glpr(**kwargs)` uses `kwargs.get("layout_det_results", [])` — matches how `_process_vlm` calls `get_layout_parsing_results(images=..., layout_det_results=..., imgs_in_doc=..., ...)` with keyword args — consistent.
