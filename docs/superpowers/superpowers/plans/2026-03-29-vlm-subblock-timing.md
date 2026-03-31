# VLM Sub-Step Timing & Parallelization Instrumentation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument `pdf_parser_service.py` with per-block-type timing, VLM sub-step breakdown, and stage 1–3 page-render timing to support parallelization analysis.

**Architecture:** All changes are confined to `pdf_parser_service.py` and its test file. A `threading.local()` accumulator (`_timing_state`) shuttles per-block data from deep VLM hooks back up to the `_timed_glpr` wrapper, which emits the aggregated `BlockType` summary and `Phases` split. Stage 1–3 timing (render/save/array) is added directly in `_run_ocr` behind the existing `_use_timing()` guard. Worker count is exposed via `PDF_VLM_WORKERS` env var (implementation deferred; config function added now).

**Tech Stack:** Python 3.13, `threading.local`, `time.perf_counter`, `pytest`, existing `_env()` / `log` helpers in `pdf_parser_service.py`.

---

## File Map

| File | Role |
|------|------|
| `pdf_parser_service.py` | All production changes |
| `tests/test_timing_hooks.py` | All test changes |

---

## Task 1: Fix Four Failing Tests in `_make_mock_engine`

**Problem:** `MagicMock()` auto-creates `_pipeline` as a *new* child mock when `getattr(outer, "_pipeline", outer)` is called, so `_install_timing_hooks` patches the child rather than `outer`. Tests that assert `pipeline.layout_det_model is _TimedModelWrapper(...)` fail because `outer` is never patched.

**Fix:** Set `pipeline._pipeline = pipeline` before the engine is returned, so the `getattr` call finds a real attribute and returns `pipeline` itself.

**Files:**
- Modify: `tests/test_timing_hooks.py` — `_make_mock_engine` only

- [ ] **Step 1.1: Run the current failing tests to capture baseline**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v 2>&1 | grep -E "PASSED|FAILED"
```

Expected: 9 passed, 4 failed.

- [ ] **Step 1.2: Apply the fix to `_make_mock_engine`**

In `tests/test_timing_hooks.py`, replace the `_make_mock_engine` body:

```python
def _make_mock_engine(self, has_preprocessor=True, has_layout=True, has_glpr=True):
    """Build a mock ocr_engine with a fake paddlex_pipeline."""
    pipeline = MagicMock()
    # _install_timing_hooks calls getattr(outer, "_pipeline", outer).
    # MagicMock auto-creates _pipeline as a *different* child object, so we
    # set it explicitly to pipeline itself so the drill-down is a no-op.
    pipeline._pipeline = pipeline
    if not has_preprocessor:
        del pipeline.doc_preprocessor_pipeline
    if not has_layout:
        del pipeline.layout_det_model
    if not has_glpr:
        del pipeline.get_layout_parsing_results
    engine = MagicMock()
    engine.paddlex_pipeline = pipeline
    return engine, pipeline
```

- [ ] **Step 1.3: Run the tests — all 13 must pass**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v
```

Expected: 13 passed, 0 failed.

- [ ] **Step 1.4: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  git add tests/test_timing_hooks.py && \
  git commit -m "test: fix _make_mock_engine so _pipeline drills down to itself"
```

---

## Task 2: Add `threading` Import, `_timing_state`, `_query_to_label`, `_vlm_workers`

These are small, independent additions. Test first, implement second.

**Files:**
- Modify: `pdf_parser_service.py`
- Modify: `tests/test_timing_hooks.py`

- [ ] **Step 2.1: Write failing tests**

Append to `tests/test_timing_hooks.py`:

```python
# ---------------------------------------------------------------------------
# _vlm_workers tests
# ---------------------------------------------------------------------------

class TestVlmWorkers:
    def setup_method(self):
        # Ensure env var is clean before each test
        os.environ.pop("PDF_VLM_WORKERS", None)

    def teardown_method(self):
        os.environ.pop("PDF_VLM_WORKERS", None)

    def test_default_is_1(self):
        from pdf_parser_service import _vlm_workers
        assert _vlm_workers() == 1

    def test_reads_env_var(self):
        os.environ["PDF_VLM_WORKERS"] = "4"
        from pdf_parser_service import _vlm_workers
        assert _vlm_workers() == 4

    def test_clamps_to_minimum_1(self):
        os.environ["PDF_VLM_WORKERS"] = "0"
        from pdf_parser_service import _vlm_workers
        assert _vlm_workers() == 1

    def test_invalid_value_returns_1(self):
        os.environ["PDF_VLM_WORKERS"] = "banana"
        from pdf_parser_service import _vlm_workers
        assert _vlm_workers() == 1


# ---------------------------------------------------------------------------
# _query_to_label tests
# ---------------------------------------------------------------------------

class TestQueryToLabel:
    def test_ocr_prefix(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("OCR:") == "ocr"

    def test_table_prefix(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("Table Recognition:") == "table"

    def test_formula_prefix(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("Formula Recognition:") == "formula"

    def test_chart_prefix(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("Chart Recognition:") == "chart"

    def test_seal_prefix(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("Seal Recognition:") == "seal"

    def test_spotting_prefix(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("Spotting:") == "spotting"

    def test_unknown_defaults_to_ocr(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("Something Else:") == "ocr"

    def test_case_insensitive(self):
        from pdf_parser_service import _query_to_label
        assert _query_to_label("table recognition:") == "table"
```

- [ ] **Step 2.2: Run — confirm new tests fail**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py::TestVlmWorkers \
    tests/test_timing_hooks.py::TestQueryToLabel -v
```

Expected: all new tests fail with `ImportError` (functions not yet defined).

- [ ] **Step 2.3: Add imports and implementations to `pdf_parser_service.py`**

In the imports block, after `import time`, add:

```python
import threading
```

After the `log = logging.getLogger(__name__)` line, add the module-level state and helpers:

```python
# ---------------------------------------------------------------------------
# Per-block timing accumulator (populated by VLM sub-step hooks, consumed by
# _timed_glpr).  threading.local so that future parallel workers stay isolated.
# ---------------------------------------------------------------------------
_timing_state = threading.local()

_QUERY_LABEL_MAP: dict[str, str] = {
    "table recognition:": "table",
    "formula recognition:": "formula",
    "chart recognition:": "chart",
    "seal recognition:": "seal",
    "spotting:": "spotting",
}


def _query_to_label(query: str) -> str:
    """Map a VLM query prefix to a short block-type label for timing logs."""
    return _QUERY_LABEL_MAP.get(query.strip().lower(), "ocr")


def _vlm_workers() -> int:
    """Number of concurrent VLM worker threads (PDF_VLM_WORKERS env var, default 1)."""
    try:
        return max(1, int(_env("PDF_VLM_WORKERS", "1")))
    except ValueError:
        return 1
```

- [ ] **Step 2.4: Run — all tests must pass**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v
```

Expected: 21 passed, 0 failed.

- [ ] **Step 2.5: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  git add pdf_parser_service.py tests/test_timing_hooks.py && \
  git commit -m "feat: add _timing_state, _query_to_label, _vlm_workers"
```

---

## Task 3: Add `_log_block_type_summary`

**Files:**
- Modify: `pdf_parser_service.py`
- Modify: `tests/test_timing_hooks.py`

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_timing_hooks.py`:

```python
# ---------------------------------------------------------------------------
# _log_block_type_summary tests
# ---------------------------------------------------------------------------

class TestLogBlockTypeSummary:
    def _blocks(self):
        return [
            {"label": "ocr",   "pixels": 100000, "pre_ms":  10.0, "xfer_ms": 1.0, "gen_ms": 2800.0, "post_ms": 1.0, "chars": 150},
            {"label": "ocr",   "pixels":  80000, "pre_ms":   8.0, "xfer_ms": 1.0, "gen_ms": 2200.0, "post_ms": 1.0, "chars":  90},
            {"label": "table", "pixels": 300000, "pre_ms":  18.0, "xfer_ms": 2.0, "gen_ms": 4000.0, "post_ms": 2.0, "chars": 600},
        ]

    def test_emits_one_line_per_unique_label(self, caplog):
        from pdf_parser_service import _log_block_type_summary
        with caplog.at_level(logging.INFO):
            _log_block_type_summary(self._blocks())
        block_type_records = [r for r in caplog.records if "BlockType" in r.message]
        labels = {r.message.split()[2].rstrip(":") for r in block_type_records}
        assert labels == {"ocr", "table"}

    def test_counts_blocks_per_label(self, caplog):
        from pdf_parser_service import _log_block_type_summary
        with caplog.at_level(logging.INFO):
            _log_block_type_summary(self._blocks())
        ocr_line = next(r.message for r in caplog.records if "BlockType" in r.message and "ocr" in r.message)
        assert "2 blocks" in ocr_line
        table_line = next(r.message for r in caplog.records if "BlockType" in r.message and "table" in r.message)
        assert "1 block" in table_line

    def test_totals_ms_per_label(self, caplog):
        from pdf_parser_service import _log_block_type_summary
        with caplog.at_level(logging.INFO):
            _log_block_type_summary(self._blocks())
        # ocr total = (10+1+2800+1) + (8+1+2200+1) = 2812 + 2210 = 5022ms
        ocr_line = next(r.message for r in caplog.records if "BlockType" in r.message and "ocr" in r.message)
        assert "5022" in ocr_line

    def test_empty_blocks_emits_nothing(self, caplog):
        from pdf_parser_service import _log_block_type_summary
        with caplog.at_level(logging.INFO):
            _log_block_type_summary([])
        assert not any("BlockType" in r.message for r in caplog.records)

    def test_sorted_by_total_ms_descending(self, caplog):
        from pdf_parser_service import _log_block_type_summary
        with caplog.at_level(logging.INFO):
            _log_block_type_summary(self._blocks())
        block_type_records = [r for r in caplog.records if "BlockType" in r.message]
        # table (4022ms total) should come before ocr (5022ms total)?
        # Actually ocr is larger (5022 vs 4022), so ocr first
        assert "ocr" in block_type_records[0].message
        assert "table" in block_type_records[1].message
```

- [ ] **Step 3.2: Run — confirm new tests fail**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py::TestLogBlockTypeSummary -v
```

Expected: all 5 fail with `ImportError`.

- [ ] **Step 3.3: Implement `_log_block_type_summary` in `pdf_parser_service.py`**

Add after `_vlm_workers()`:

```python
def _log_block_type_summary(blocks: list) -> None:
    """Emit one [TIMING] BlockType log line per label, sorted by total ms descending."""
    if not blocks:
        return
    totals: dict[str, dict] = {}
    for b in blocks:
        label = b["label"]
        if label not in totals:
            totals[label] = {"count": 0, "total_ms": 0.0, "total_px": 0, "total_chars": 0}
        t = totals[label]
        t["count"] += 1
        t["total_ms"] += b["pre_ms"] + b["xfer_ms"] + b["gen_ms"] + b["post_ms"]
        t["total_px"] += b["pixels"]
        t["total_chars"] += b["chars"]
    for label, t in sorted(totals.items(), key=lambda x: -x[1]["total_ms"]):
        n = t["count"]
        avg_ms = t["total_ms"] / n
        avg_px_k = t["total_px"] / n / 1000
        avg_chars = t["total_chars"] // n
        log.info(
            "[TIMING] BlockType %-22s %2d %s  avg=%5.0fms  avg_px=%5.1fK  avg_chars=%4d  total=%7.0fms",
            label + ":",
            n,
            "blocks" if n != 1 else "block ",
            avg_ms,
            avg_px_k,
            avg_chars,
            t["total_ms"],
        )
```

- [ ] **Step 3.4: Run — all tests pass**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v
```

Expected: 26 passed, 0 failed.

- [ ] **Step 3.5: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  git add pdf_parser_service.py tests/test_timing_hooks.py && \
  git commit -m "feat: add _log_block_type_summary helper"
```

---

## Task 4: Add Stage 1–3 Timing in `_run_ocr`

**Files:**
- Modify: `pdf_parser_service.py` — `_run_ocr` function
- Modify: `tests/test_timing_hooks.py` — add `TestRunOcrTiming`

- [ ] **Step 4.1: Write failing test**

Append to `tests/test_timing_hooks.py`:

```python
# ---------------------------------------------------------------------------
# _run_ocr stage 1-3 timing tests
# ---------------------------------------------------------------------------

class TestRunOcrTiming:
    def setup_method(self):
        os.environ["PDF_TIMING"] = "true"

    def teardown_method(self):
        os.environ.pop("PDF_TIMING", None)

    def _make_mocks(self, tmp_path):
        """Return (mock_engine, mock_doc, mock_page, mock_pix)."""
        import sys

        mock_pix = MagicMock()
        mock_pix.height = 100
        mock_pix.width = 80
        # samples must be bytes-like; numpy is mocked so frombuffer just returns a MagicMock
        mock_pix.samples = b"\x00" * (100 * 80 * 3)

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        fitz_mod = sys.modules["fitz"]
        fitz_mod.open.return_value = mock_doc
        fitz_mod.Matrix.return_value = MagicMock()
        fitz_mod.csRGB = MagicMock()

        mock_engine = MagicMock()
        mock_engine.predict.return_value = []

        return mock_engine, mock_doc, mock_page, mock_pix

    def test_logs_page_render_timing(self, caplog, tmp_path):
        from pdf_parser_service import _run_ocr
        mock_engine, *_ = self._make_mocks(tmp_path)
        with caplog.at_level(logging.INFO):
            _run_ocr(mock_engine, "fake.pdf", str(tmp_path), lambda s, p: None)
        assert any("PageRender" in r.message and "ms" in r.message
                   for r in caplog.records)

    def test_logs_png_save_timing(self, caplog, tmp_path):
        from pdf_parser_service import _run_ocr
        mock_engine, *_ = self._make_mocks(tmp_path)
        with caplog.at_level(logging.INFO):
            _run_ocr(mock_engine, "fake.pdf", str(tmp_path), lambda s, p: None)
        assert any("PNGSave" in r.message and "ms" in r.message
                   for r in caplog.records)

    def test_logs_array_build_timing(self, caplog, tmp_path):
        from pdf_parser_service import _run_ocr
        mock_engine, *_ = self._make_mocks(tmp_path)
        with caplog.at_level(logging.INFO):
            _run_ocr(mock_engine, "fake.pdf", str(tmp_path), lambda s, p: None)
        assert any("ArrayBuild" in r.message and "ms" in r.message
                   for r in caplog.records)

    def test_no_timing_logs_when_pdf_timing_false(self, caplog, tmp_path):
        os.environ["PDF_TIMING"] = "false"
        from pdf_parser_service import _run_ocr
        mock_engine, *_ = self._make_mocks(tmp_path)
        with caplog.at_level(logging.INFO):
            _run_ocr(mock_engine, "fake.pdf", str(tmp_path), lambda s, p: None)
        timing_stage_msgs = [r.message for r in caplog.records
                             if any(k in r.message for k in ("PageRender", "PNGSave", "ArrayBuild"))]
        assert timing_stage_msgs == []
```

- [ ] **Step 4.2: Run — confirm new tests fail**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py::TestRunOcrTiming -v
```

Expected: all 4 fail.

- [ ] **Step 4.3: Add stage 1–3 timing guards in `_run_ocr`**

Locate the `_run_ocr` function in `pdf_parser_service.py`. Find the block that starts with `pix = page.get_pixmap(...)` inside the `for page_num` loop. Replace:

```python
            # Render to RGB numpy array — avoids writing a PNG to disk.
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
            image_filename = f"page_{page_num + 1}.png"
            image_path = os.path.join(work_dir, image_filename)
            pix.save(image_path)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
```

With:

```python
            # Render to RGB numpy array — avoids writing a PNG to disk.
            if _use_timing():
                _t_render = time.perf_counter()
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
            image_filename = f"page_{page_num + 1}.png"
            image_path = os.path.join(work_dir, image_filename)
            if _use_timing():
                _t_save = time.perf_counter()
                log.info("[TIMING] PageRender: %.0fms", (_t_save - _t_render) * 1000)
            pix.save(image_path)
            if _use_timing():
                _t_array = time.perf_counter()
                log.info("[TIMING] PNGSave: %.0fms", (_t_array - _t_save) * 1000)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            if _use_timing():
                log.info("[TIMING] ArrayBuild: %.0fms", (time.perf_counter() - _t_array) * 1000)
```

- [ ] **Step 4.4: Run — all tests pass**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v
```

Expected: 30 passed, 0 failed.

- [ ] **Step 4.5: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  git add pdf_parser_service.py tests/test_timing_hooks.py && \
  git commit -m "feat: add PageRender/PNGSave/ArrayBuild timing to _run_ocr"
```

---

## Task 5: Wrap `vl_rec_model` Sub-Steps in `_install_timing_hooks`

This extends `_install_timing_hooks` with a new section that wraps `vl_rec_model.process` (for per-block metadata + log line) and the four sub-objects for sub-step timing.

**Files:**
- Modify: `pdf_parser_service.py` — extend `_install_timing_hooks`
- Modify: `tests/test_timing_hooks.py` — add three new tests to `TestInstallTimingHooks`

- [ ] **Step 5.1: Write failing tests**

Append the following three methods inside the `TestInstallTimingHooks` class:

```python
    def test_vl_rec_model_process_is_replaced_with_wrapper(self):
        from pdf_parser_service import _install_timing_hooks
        engine, pipeline = self._make_mock_engine()
        original_process = pipeline.vl_rec_model.process
        _install_timing_hooks(engine)
        assert pipeline.vl_rec_model.process is not original_process
        assert callable(pipeline.vl_rec_model.process)

    def test_vl_rec_model_process_populates_timing_state(self):
        from pdf_parser_service import _install_timing_hooks, _timing_state
        engine, pipeline = self._make_mock_engine()
        pipeline.vl_rec_model.process.return_value = {"result": ["hello world"]}
        _install_timing_hooks(engine)

        _timing_state.blocks = []
        _timing_state.vlm_total_ms = 0.0

        mock_img = MagicMock()
        mock_img.shape = (100, 80, 3)
        pipeline.vl_rec_model.process([{"image": mock_img, "query": "OCR:"}])

        assert len(_timing_state.blocks) == 1
        block = _timing_state.blocks[0]
        assert block["label"] == "ocr"
        assert block["pixels"] == 8000
        assert block["chars"] == 11   # len("hello world")
        assert "pre_ms" in block and "gen_ms" in block

    def test_vl_rec_model_process_logs_block_line(self, caplog):
        from pdf_parser_service import _install_timing_hooks, _timing_state
        engine, pipeline = self._make_mock_engine()
        pipeline.vl_rec_model.process.return_value = {"result": ["result text"]}
        _install_timing_hooks(engine)

        _timing_state.blocks = []
        _timing_state.vlm_total_ms = 0.0

        mock_img = MagicMock()
        mock_img.shape = (200, 150, 3)
        with caplog.at_level(logging.INFO):
            pipeline.vl_rec_model.process([{"image": mock_img, "query": "Table Recognition:"}])

        block_records = [r for r in caplog.records if "[TIMING] Block" in r.message]
        assert len(block_records) == 1
        assert "table" in block_records[0].message
        assert "30000" in block_records[0].message   # pixels = 200*150
```

- [ ] **Step 5.2: Run — confirm new tests fail**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest \
    tests/test_timing_hooks.py::TestInstallTimingHooks::test_vl_rec_model_process_is_replaced_with_wrapper \
    tests/test_timing_hooks.py::TestInstallTimingHooks::test_vl_rec_model_process_populates_timing_state \
    tests/test_timing_hooks.py::TestInstallTimingHooks::test_vl_rec_model_process_logs_block_line \
    -v
```

Expected: all 3 fail.

- [ ] **Step 5.3: Extend `_install_timing_hooks` with `vl_rec_model` hooks**

In `pdf_parser_service.py`, inside `_install_timing_hooks`, append the following block **after** the existing `get_layout_parsing_results` patch:

```python
    # ------------------------------------------------------------------
    # vl_rec_model sub-step hooks
    # ------------------------------------------------------------------
    if not hasattr(pipeline, "vl_rec_model"):
        return

    vl_model = pipeline.vl_rec_model

    # Wrap each sub-step.  Each wrapper stores its elapsed ms into
    # _timing_state.current_block (a dict set up by _timed_process below).
    # We guard with hasattr so the function is silent if PaddleOCR internals change.

    if hasattr(vl_model, "processor") and hasattr(vl_model.processor, "preprocess"):
        _orig_pre = vl_model.processor.preprocess

        def _timed_preprocess(*args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_pre(*args, **kwargs)
            if hasattr(_timing_state, "current_block"):
                _timing_state.current_block["pre_ms"] = (time.perf_counter() - t0) * 1000
            return result

        vl_model.processor.preprocess = _timed_preprocess

    if hasattr(vl_model, "_switch_inputs_to_device"):
        _orig_xfer = vl_model._switch_inputs_to_device

        def _timed_xfer(*args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_xfer(*args, **kwargs)
            if hasattr(_timing_state, "current_block"):
                _timing_state.current_block["xfer_ms"] = (time.perf_counter() - t0) * 1000
            return result

        vl_model._switch_inputs_to_device = _timed_xfer

    if hasattr(vl_model, "infer") and hasattr(vl_model.infer, "generate"):
        _orig_gen = vl_model.infer.generate

        def _timed_generate(*args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_gen(*args, **kwargs)
            if hasattr(_timing_state, "current_block"):
                _timing_state.current_block["gen_ms"] = (time.perf_counter() - t0) * 1000
            return result

        vl_model.infer.generate = _timed_generate

    if hasattr(vl_model, "processor") and hasattr(vl_model.processor, "postprocess"):
        _orig_post = vl_model.processor.postprocess

        def _timed_postprocess(*args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_post(*args, **kwargs)
            if hasattr(_timing_state, "current_block"):
                _timing_state.current_block["post_ms"] = (time.perf_counter() - t0) * 1000
            return result

        vl_model.processor.postprocess = _timed_postprocess

    if hasattr(vl_model, "process"):
        _orig_process = vl_model.process

        def _timed_process(data, **kwargs):
            # Extract input metadata
            first = data[0] if data else {}
            img = first.get("image")
            query = first.get("query", "")
            try:
                pixels = img.shape[0] * img.shape[1]
            except (AttributeError, TypeError, IndexError):
                pixels = 0
            label = _query_to_label(query)

            # Reset sub-step accumulator for this block
            _timing_state.current_block = {
                "pre_ms": 0.0, "xfer_ms": 0.0, "gen_ms": 0.0, "post_ms": 0.0,
            }

            result = _orig_process(data, **kwargs)

            # Extract output char count
            try:
                chars = sum(
                    len(r) for r in result.get("result", []) if isinstance(r, str)
                )
            except (AttributeError, TypeError):
                chars = 0

            block = dict(_timing_state.current_block)
            block.update({"label": label, "pixels": pixels, "chars": chars})
            total_block_ms = block["pre_ms"] + block["xfer_ms"] + block["gen_ms"] + block["post_ms"]

            log.info(
                "[TIMING] Block label=%-22s pixels=%7d"
                "  pre=%5.0fms xfer=%4.0fms gen=%6.0fms post=%4.0fms chars=%5d",
                label, pixels,
                block["pre_ms"], block["xfer_ms"], block["gen_ms"], block["post_ms"], chars,
            )

            if not hasattr(_timing_state, "blocks"):
                _timing_state.blocks = []
            if not hasattr(_timing_state, "vlm_total_ms"):
                _timing_state.vlm_total_ms = 0.0
            _timing_state.blocks.append(block)
            _timing_state.vlm_total_ms += total_block_ms

            return result

        vl_model.process = _timed_process
```

- [ ] **Step 5.4: Run — all tests pass**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v
```

Expected: 33 passed, 0 failed.

- [ ] **Step 5.5: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  git add pdf_parser_service.py tests/test_timing_hooks.py && \
  git commit -m "feat: wrap vl_rec_model sub-steps for per-block timing"
```

---

## Task 6: Update `_timed_glpr` for Block-Type Aggregation and Phases Split

**Files:**
- Modify: `pdf_parser_service.py` — `_timed_glpr` closure inside `_install_timing_hooks`
- Modify: `tests/test_timing_hooks.py` — update one existing test, add two new tests

- [ ] **Step 6.1: Write failing tests**

Append three new methods to `TestInstallTimingHooks`:

```python
    def test_glpr_initializes_timing_state_on_each_call(self):
        from pdf_parser_service import _install_timing_hooks, _timing_state
        engine, pipeline = self._make_mock_engine()
        pipeline.get_layout_parsing_results.return_value = None
        _install_timing_hooks(engine)

        # Pre-populate with stale data from a previous call
        _timing_state.blocks = [{"label": "stale"}]
        _timing_state.vlm_total_ms = 999.0

        pipeline.get_layout_parsing_results(
            layout_det_results=[], images=[], imgs_in_doc=[]
        )

        # Must be reset to empty list (stale block gone)
        assert _timing_state.blocks == []
        assert _timing_state.vlm_total_ms == 0.0

    def test_glpr_logs_block_type_summary_when_blocks_present(self, caplog):
        from pdf_parser_service import _install_timing_hooks, _timing_state
        engine, pipeline = self._make_mock_engine()

        # Use side_effect so the original glpr mock populates _timing_state.blocks
        # *during* the _timed_glpr call (after the reset but before the summary).
        # Pre-populating before the call would be wiped by _timed_glpr's reset.
        def _glpr_side_effect(*args, **kwargs):
            _timing_state.blocks.append({
                "label": "ocr", "pixels": 90000,
                "pre_ms": 10.0, "xfer_ms": 1.0, "gen_ms": 2500.0, "post_ms": 1.0, "chars": 100,
            })
            _timing_state.blocks.append({
                "label": "table", "pixels": 200000,
                "pre_ms": 15.0, "xfer_ms": 2.0, "gen_ms": 3800.0, "post_ms": 2.0, "chars": 500,
            })
            _timing_state.vlm_total_ms = 6331.0
            return None

        pipeline.get_layout_parsing_results.side_effect = _glpr_side_effect
        _install_timing_hooks(engine)

        with caplog.at_level(logging.INFO):
            pipeline.get_layout_parsing_results(
                layout_det_results=[], images=[], imgs_in_doc=[]
            )

        block_type_records = [r for r in caplog.records if "BlockType" in r.message]
        block_type_labels = {r.message.split()[2].rstrip(":") for r in block_type_records}
        assert "ocr" in block_type_labels
        assert "table" in block_type_labels

    def test_glpr_logs_phases_split(self, caplog):
        from pdf_parser_service import _install_timing_hooks, _timing_state
        engine, pipeline = self._make_mock_engine()
        pipeline.get_layout_parsing_results.return_value = None
        _install_timing_hooks(engine)

        _timing_state.blocks = []
        _timing_state.vlm_total_ms = 0.0

        with caplog.at_level(logging.INFO):
            pipeline.get_layout_parsing_results(
                layout_det_results=[{"boxes": [{"label": "text"}]}],
                images=[], imgs_in_doc=[],
            )

        phases_records = [r for r in caplog.records if "Phases:" in r.message]
        assert len(phases_records) == 1
        assert "vlm_loop=" in phases_records[0].message
        assert "other=" in phases_records[0].message
        assert "total=" in phases_records[0].message
```

Also update the existing `test_glpr_closure_logs_block_count_and_timing` to check the `VLM:` log still appears (it should — we keep it):

```python
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
        assert "3 blocks" in vlm_log.message
```

- [ ] **Step 6.2: Run — confirm new tests fail**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest \
    tests/test_timing_hooks.py::TestInstallTimingHooks::test_glpr_initializes_timing_state_on_each_call \
    tests/test_timing_hooks.py::TestInstallTimingHooks::test_glpr_logs_block_type_summary_when_blocks_present \
    tests/test_timing_hooks.py::TestInstallTimingHooks::test_glpr_logs_phases_split \
    -v
```

Expected: all 3 fail.

- [ ] **Step 6.3: Update `_timed_glpr` closure inside `_install_timing_hooks`**

Replace the existing `_timed_glpr` definition in `_install_timing_hooks` with:

```python
        def _timed_glpr(*args, **kwargs):
            det_results = kwargs.get("layout_det_results", [])
            label_counts = Counter(
                b.get("label", "?")
                for r in det_results
                for b in r.get("boxes", [])
            )
            n_blocks = sum(label_counts.values())
            log.info("[TIMING] Blocks: %s", dict(label_counts))

            # Reset per-block accumulator for this page
            _timing_state.blocks = []
            _timing_state.vlm_total_ms = 0.0

            t0 = time.perf_counter()
            result = _orig_glpr(*args, **kwargs)
            total_ms = (time.perf_counter() - t0) * 1000

            # Emit per-block-type summary (populated by _timed_process hooks)
            _log_block_type_summary(_timing_state.blocks)

            # Phases split: vlm_loop is the sum of all _timed_process durations;
            # other covers block prep + result assembly (not separately measurable
            # without modifying pipeline source)
            vlm_ms = _timing_state.vlm_total_ms
            other_ms = total_ms - vlm_ms
            log.info(
                "[TIMING] Phases: vlm_loop=%.0fms  other=%.0fms  total=%.0fms",
                vlm_ms, other_ms, total_ms,
            )

            if n_blocks:
                log.info(
                    "[TIMING] VLM: %d blocks, %.0fms total, %.0fms avg/block",
                    n_blocks, total_ms, total_ms / n_blocks,
                )
            else:
                log.info("[TIMING] VLM: 0 blocks, %.0fms total", total_ms)
            return result
```

- [ ] **Step 6.4: Run — all tests pass**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v
```

Expected: 36 passed, 0 failed.

- [ ] **Step 6.5: Commit**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  git add pdf_parser_service.py tests/test_timing_hooks.py && \
  git commit -m "feat: emit BlockType summary and Phases split in _timed_glpr"
```

---

## Final Verification

- [ ] **Run full test suite one last time**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  .venv/bin/python -m pytest tests/test_timing_hooks.py -v
```

Expected: 36 passed, 0 failed.

- [ ] **Smoke-check the docstring at top of `_install_timing_hooks`** — update the "Log output per page" example comment to reflect the new log lines:

```python
    """...
    Log output per page (example):
        [TIMING] PageRender:   38ms
        [TIMING] PNGSave:      22ms
        [TIMING] ArrayBuild:    2ms
        [TIMING] DocPreproc:   12ms          (only when doc_preprocessor enabled)
        [TIMING] LayoutDet:   430ms, boxes=14
        [TIMING] Blocks: {'text': 9, 'table': 1, 'formula': 3, 'title': 1}
        [TIMING] Block label=ocr            pixels=142080  pre=   11ms xfer=   1ms gen= 2841ms post=   1ms chars=  183
        [TIMING] Block label=table          pixels=318400  pre=   18ms xfer=   2ms gen= 4120ms post=   2ms chars=  612
        [TIMING] BlockType ocr:             9 blocks  avg= 2950ms  avg_px= 120.0K  avg_chars= 150  total= 26550ms
        [TIMING] BlockType table:           1 block   avg= 4120ms  avg_px= 318.4K  avg_chars= 612  total=  4120ms
        [TIMING] Phases: vlm_loop=30670ms  other=215ms  total=30885ms
        [TIMING] VLM: 14 blocks, 30885ms total, 2206ms avg/block
    ...
    """
```

- [ ] **Commit docstring update**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr && \
  git add pdf_parser_service.py && \
  git commit -m "docs: update _install_timing_hooks docstring with Phase 2 log format"
```

---

## Notes for Implementer

**Why query-derived labels instead of exact layout-detection labels?**
`DocVLMPredictor.process()` only receives a cropped image and query string — the original layout-detection block label (e.g., `header`, `footer`, `number`) is not passed through. The query prefix (`"OCR:"`) groups all such block types under `"ocr"` in the per-block log. The existing `[TIMING] Blocks: {...}` line (unchanged) still shows the full label breakdown from layout detection; the new `BlockType` summary shows which *query type* dominates VLM time.

**`PDF_VLM_WORKERS` is config-only in this phase.**
The `_vlm_workers()` function is added now so env-var parsing and clamping logic is in place. Actual parallel execution (ThreadPoolExecutor or queue-based) is the next implementation phase after timing data is collected and the parallelization decision is made.

**Closing the timing gap: `other` in `Phases`.**
`other_ms = total_ms - vlm_loop_ms` covers both block prep (filter/crop/merge/group) and result assembly (OTSL→HTML, table untokenize). These cannot be separated without modifying `.venv/…/pipeline.py`. If the data shows `other` is non-trivial (>200ms), a follow-up can instrument those sub-phases by adding a thin shim layer over `get_layout_parsing_results` inputs.
