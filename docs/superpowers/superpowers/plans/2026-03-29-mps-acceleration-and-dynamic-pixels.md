# MPS Acceleration & Dynamic min_pixels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-page VLM inference time ~12× by moving the PyTorch model to Apple Metal GPU and capping visual token count for small OCR blocks.

**Architecture:** Two monkey-patch functions (`_enable_mps_acceleration`, `_install_dynamic_pixels_hook`) applied to the live `DocVLMPredictor` instance after engine init in `run()`. No installed package files are modified. Timing hooks are applied last so they measure the real post-optimization cost.

**Tech Stack:** Python 3.13, PyTorch (MPS backend), `unittest.mock`, `pytest`

---

## File Map

| File | Role |
|---|---|
| `pdf_parser_service.py` | All new code lives here — helpers, two new functions, `run()` wiring |
| `tests/test_timing_hooks.py` | All new tests live here — two new test classes |

---

## Task 1: Config Helpers — `_use_mps` and `_vlm_ocr_max_pixels`

**Files:**
- Modify: `pdf_parser_service.py` (near other `_use_*` helpers, around line 169)
- Modify: `tests/test_timing_hooks.py` (new `TestConfigHelpers` class at end)

- [ ] **Step 1: Write the failing tests**

Add this class to `tests/test_timing_hooks.py` (after the existing `TestVlmWorkers` class):

```python
# ---------------------------------------------------------------------------
# _use_mps / _vlm_ocr_max_pixels tests
# ---------------------------------------------------------------------------

class TestConfigHelpers:
    def setup_method(self):
        os.environ.pop("PDF_MPS", None)
        os.environ.pop("PDF_VLM_OCR_MAX_PIXELS", None)

    def teardown_method(self):
        os.environ.pop("PDF_MPS", None)
        os.environ.pop("PDF_VLM_OCR_MAX_PIXELS", None)

    def test_use_mps_default_false(self):
        from pdf_parser_service import _use_mps
        assert _use_mps() is False

    def test_use_mps_env_true(self):
        os.environ["PDF_MPS"] = "true"
        from pdf_parser_service import _use_mps
        assert _use_mps() is True

    def test_use_mps_case_insensitive(self):
        os.environ["PDF_MPS"] = "TRUE"
        from pdf_parser_service import _use_mps
        assert _use_mps() is True

    def test_vlm_ocr_max_pixels_default(self):
        from pdf_parser_service import _vlm_ocr_max_pixels
        assert _vlm_ocr_max_pixels() == 50176

    def test_vlm_ocr_max_pixels_env_override(self):
        os.environ["PDF_VLM_OCR_MAX_PIXELS"] = "28224"
        from pdf_parser_service import _vlm_ocr_max_pixels
        assert _vlm_ocr_max_pixels() == 28224

    def test_vlm_ocr_max_pixels_floor_enforced(self):
        os.environ["PDF_VLM_OCR_MAX_PIXELS"] = "100"  # below floor of 12544
        from pdf_parser_service import _vlm_ocr_max_pixels
        assert _vlm_ocr_max_pixels() == 12544

    def test_vlm_ocr_max_pixels_invalid_returns_default(self):
        os.environ["PDF_VLM_OCR_MAX_PIXELS"] = "banana"
        from pdf_parser_service import _vlm_ocr_max_pixels
        assert _vlm_ocr_max_pixels() == 50176
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /Users/cding/Workspace/ThirdParty/paddleocr
.venv/bin/python -m pytest tests/test_timing_hooks.py::TestConfigHelpers -v 2>&1 | tail -15
```

Expected: `ERROR` — `_use_mps` not defined.

- [ ] **Step 3: Add helpers to `pdf_parser_service.py`**

Insert after the `_use_timing` function (around line 170):

```python
def _use_mps() -> bool:
    return _env("PDF_MPS", "false").lower() == "true"


def _vlm_ocr_max_pixels() -> int:
    """Cap on min_pixels used for OCR blocks. Default 50176 (224²).
    Floor of 12544 (112²) prevents values too small to read text."""
    try:
        return max(12544, int(_env("PDF_VLM_OCR_MAX_PIXELS", "50176")))
    except ValueError:
        return 50176
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py::TestConfigHelpers -v 2>&1 | tail -15
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add pdf_parser_service.py tests/test_timing_hooks.py
git commit -m "feat: add _use_mps and _vlm_ocr_max_pixels config helpers"
```

---

## Task 2: MPS Acceleration — `_enable_mps_acceleration`

**Files:**
- Modify: `pdf_parser_service.py` (new function after `_install_timing_hooks`)
- Modify: `tests/test_timing_hooks.py` (new `TestEnableMpsAcceleration` class)

- [ ] **Step 1: Write the failing tests**

Add this helper and class to `tests/test_timing_hooks.py` (after `TestConfigHelpers`):

```python
# ---------------------------------------------------------------------------
# _enable_mps_acceleration tests
# ---------------------------------------------------------------------------

def _make_mock_torch(mps_available=True, to_raises=False):
    """Minimal torch mock. Defines a real FakeTensor class so isinstance() works."""
    class FakeTensor:
        def __init__(self):
            self._moved_to = None
        def to(self, device):
            self._moved_to = device
            return self

    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = mps_available
    mock_torch.Tensor = FakeTensor
    if to_raises:
        mock_torch.backends.mps.is_available.return_value = True  # still available
    return mock_torch, FakeTensor


class TestEnableMpsAcceleration:
    def _make_mock_engine(self):
        pipeline = MagicMock()
        pipeline._pipeline = pipeline
        vl_model = MagicMock()
        vl_model._mps_enabled = False
        pipeline.vl_rec_model = vl_model
        engine = MagicMock()
        engine.paddlex_pipeline = pipeline
        return engine, pipeline, vl_model

    def test_mps_moves_model_to_mps(self):
        import sys
        engine, pipeline, vl_model = self._make_mock_engine()
        mock_torch, _ = _make_mock_torch(mps_available=True)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from pdf_parser_service import _enable_mps_acceleration
            _enable_mps_acceleration(engine)

        vl_model.infer.to.assert_called_once_with("mps")

    def test_mps_patches_switch_inputs(self):
        import sys
        engine, pipeline, vl_model = self._make_mock_engine()
        mock_torch, FakeTensor = _make_mock_torch(mps_available=True)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from pdf_parser_service import _enable_mps_acceleration
            _enable_mps_acceleration(engine)

        # _switch_inputs_to_device is now the closure
        fake_tensor = FakeTensor()
        result = vl_model._switch_inputs_to_device({"img": fake_tensor, "text": "hello"})
        assert result["img"]._moved_to == "mps"

    def test_mps_skips_non_tensor_values(self):
        import sys
        engine, pipeline, vl_model = self._make_mock_engine()
        mock_torch, FakeTensor = _make_mock_torch(mps_available=True)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from pdf_parser_service import _enable_mps_acceleration
            _enable_mps_acceleration(engine)

        result = vl_model._switch_inputs_to_device({"text": "hello", "num": 42})
        assert result["text"] == "hello"
        assert result["num"] == 42

    def test_mps_not_available_is_noop(self, caplog):
        import sys
        engine, pipeline, vl_model = self._make_mock_engine()
        mock_torch, _ = _make_mock_torch(mps_available=False)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from pdf_parser_service import _enable_mps_acceleration
            with caplog.at_level(logging.WARNING):
                _enable_mps_acceleration(engine)

        vl_model.infer.to.assert_not_called()
        assert any("not available" in r.message for r in caplog.records)

    def test_mps_to_raises_is_noop(self, caplog):
        import sys
        engine, pipeline, vl_model = self._make_mock_engine()
        mock_torch, _ = _make_mock_torch(mps_available=True)
        vl_model.infer.to.side_effect = RuntimeError("unsupported op")

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from pdf_parser_service import _enable_mps_acceleration
            with caplog.at_level(logging.WARNING):
                _enable_mps_acceleration(engine)

        # _switch_inputs_to_device should NOT have been patched
        assert not getattr(vl_model, "_mps_enabled", False)
        assert any("failed" in r.message for r in caplog.records)

    def test_mps_double_install_skipped(self, caplog):
        import sys
        engine, pipeline, vl_model = self._make_mock_engine()
        mock_torch, _ = _make_mock_torch(mps_available=True)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from pdf_parser_service import _enable_mps_acceleration
            _enable_mps_acceleration(engine)
            first_switch = vl_model._switch_inputs_to_device
            with caplog.at_level(logging.WARNING):
                _enable_mps_acceleration(engine)  # second call

        assert vl_model._switch_inputs_to_device is first_switch  # not replaced again
        assert any("already enabled" in r.message for r in caplog.records)

    def test_mps_sets_enabled_flag(self):
        import sys
        engine, pipeline, vl_model = self._make_mock_engine()
        mock_torch, _ = _make_mock_torch(mps_available=True)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from pdf_parser_service import _enable_mps_acceleration
            _enable_mps_acceleration(engine)

        assert vl_model._mps_enabled is True
```

Also add `from unittest.mock import patch` to the existing imports at the top of the test file (it currently only has `from unittest.mock import MagicMock`).

- [ ] **Step 2: Run to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py::TestEnableMpsAcceleration -v 2>&1 | tail -20
```

Expected: `ERROR` — `_enable_mps_acceleration` not defined.

- [ ] **Step 3: Add `_enable_mps_acceleration` to `pdf_parser_service.py`**

Insert after `_install_timing_hooks` (after line ~541):

```python
def _enable_mps_acceleration(ocr_engine) -> None:
    """Move the VLM PyTorch model to Apple Metal GPU (MPS).

    Patches two things on the live DocVLMPredictor instance:
      1. vl_model.infer        — moved to MPS via .to("mps")
      2. _switch_inputs_to_device — replaced to move torch.Tensor inputs to MPS

    On any failure (MPS unavailable, unsupported op), logs a warning and
    leaves the model on CPU — no crash.
    """
    import torch

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        log.warning("[MPS] not available — model stays on CPU")
        return

    outer = ocr_engine.paddlex_pipeline
    pipeline = getattr(outer, "_pipeline", outer)

    if not hasattr(pipeline, "vl_rec_model"):
        log.warning("[MPS] vl_rec_model not found — skipping")
        return

    vl_model = pipeline.vl_rec_model

    if getattr(vl_model, "_mps_enabled", False):
        log.warning("[MPS] already enabled — skipping")
        return

    try:
        vl_model.infer = vl_model.infer.to("mps")
    except Exception as exc:
        log.warning("[MPS] failed to move model to MPS: %s — staying on CPU", exc)
        return

    def _mps_switch_inputs(input_dict):
        return {
            k: v.to("mps") if isinstance(v, torch.Tensor) else v
            for k, v in input_dict.items()
        }

    vl_model._switch_inputs_to_device = _mps_switch_inputs
    vl_model._mps_enabled = True
    log.info("[MPS] VLM model and input routing moved to Metal GPU")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py::TestEnableMpsAcceleration -v 2>&1 | tail -20
```

Expected: 7 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pdf_parser_service.py tests/test_timing_hooks.py
git commit -m "feat: add _enable_mps_acceleration to move VLM to Metal GPU"
```

---

## Task 3: Dynamic Pixels Hook — `_install_dynamic_pixels_hook`

**Files:**
- Modify: `pdf_parser_service.py` (new function after `_enable_mps_acceleration`)
- Modify: `tests/test_timing_hooks.py` (new `TestDynamicPixelsHook` class)

- [ ] **Step 1: Write the failing tests**

Add this class to `tests/test_timing_hooks.py` (after `TestEnableMpsAcceleration`):

```python
# ---------------------------------------------------------------------------
# _install_dynamic_pixels_hook tests
# ---------------------------------------------------------------------------

class TestDynamicPixelsHook:
    def _make_mock_engine(self):
        pipeline = MagicMock()
        pipeline._pipeline = pipeline
        vl_model = MagicMock()
        vl_model._dynamic_pixels_installed = False
        pipeline.vl_rec_model = vl_model
        engine = MagicMock()
        engine.paddlex_pipeline = pipeline
        return engine, pipeline, vl_model

    def test_ocr_block_min_pixels_is_capped(self):
        from pdf_parser_service import _install_dynamic_pixels_hook
        engine, pipeline, vl_model = self._make_mock_engine()
        original_process = vl_model.process
        original_process.return_value = {"result": []}

        _install_dynamic_pixels_hook(engine)
        # pipeline default: 112896; cap default: 50176
        vl_model.process([{"image": MagicMock(), "query": "OCR:"}], min_pixels=112896)

        original_process.assert_called_once_with(
            [{"image": original_process.call_args[0][0][0]["image"], "query": "OCR:"}],
            min_pixels=50176,
        )

    def test_non_ocr_block_min_pixels_unchanged(self):
        from pdf_parser_service import _install_dynamic_pixels_hook
        engine, pipeline, vl_model = self._make_mock_engine()
        original_process = vl_model.process
        original_process.return_value = {"result": []}

        _install_dynamic_pixels_hook(engine)
        vl_model.process(
            [{"image": MagicMock(), "query": "Table Recognition:"}],
            min_pixels=112896,
        )

        _, kwargs = original_process.call_args
        assert kwargs["min_pixels"] == 112896  # not capped

    def test_min_pixels_already_below_cap_unchanged(self):
        from pdf_parser_service import _install_dynamic_pixels_hook
        engine, pipeline, vl_model = self._make_mock_engine()
        original_process = vl_model.process
        original_process.return_value = {"result": []}

        _install_dynamic_pixels_hook(engine)
        vl_model.process([{"image": MagicMock(), "query": "OCR:"}], min_pixels=28224)

        _, kwargs = original_process.call_args
        assert kwargs["min_pixels"] == 28224  # 28224 < 50176 cap, so unchanged

    def test_min_pixels_none_unchanged(self):
        from pdf_parser_service import _install_dynamic_pixels_hook
        engine, pipeline, vl_model = self._make_mock_engine()
        original_process = vl_model.process
        original_process.return_value = {"result": []}

        _install_dynamic_pixels_hook(engine)
        vl_model.process([{"image": MagicMock(), "query": "OCR:"}], min_pixels=None)

        _, kwargs = original_process.call_args
        assert kwargs["min_pixels"] is None

    def test_formula_block_min_pixels_unchanged(self):
        from pdf_parser_service import _install_dynamic_pixels_hook
        engine, pipeline, vl_model = self._make_mock_engine()
        original_process = vl_model.process
        original_process.return_value = {"result": []}

        _install_dynamic_pixels_hook(engine)
        vl_model.process(
            [{"image": MagicMock(), "query": "Formula Recognition:"}],
            min_pixels=112896,
        )

        _, kwargs = original_process.call_args
        assert kwargs["min_pixels"] == 112896

    def test_env_var_cap_is_respected(self):
        os.environ["PDF_VLM_OCR_MAX_PIXELS"] = "28224"
        try:
            from pdf_parser_service import _install_dynamic_pixels_hook
            engine, pipeline, vl_model = self._make_mock_engine()
            original_process = vl_model.process
            original_process.return_value = {"result": []}

            _install_dynamic_pixels_hook(engine)
            vl_model.process([{"image": MagicMock(), "query": "OCR:"}], min_pixels=112896)

            _, kwargs = original_process.call_args
            assert kwargs["min_pixels"] == 28224
        finally:
            os.environ.pop("PDF_VLM_OCR_MAX_PIXELS", None)

    def test_double_install_skipped(self, caplog):
        from pdf_parser_service import _install_dynamic_pixels_hook
        engine, pipeline, vl_model = self._make_mock_engine()
        vl_model.process.return_value = {"result": []}

        _install_dynamic_pixels_hook(engine)
        first_process = vl_model.process  # the closure
        with caplog.at_level(logging.WARNING):
            _install_dynamic_pixels_hook(engine)  # second call

        assert vl_model.process is first_process  # not replaced again
        assert any("already installed" in r.message for r in caplog.records)

    def test_sets_installed_flag(self):
        from pdf_parser_service import _install_dynamic_pixels_hook
        engine, pipeline, vl_model = self._make_mock_engine()
        vl_model.process.return_value = {"result": []}

        _install_dynamic_pixels_hook(engine)

        assert vl_model._dynamic_pixels_installed is True
```

- [ ] **Step 2: Run to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py::TestDynamicPixelsHook -v 2>&1 | tail -20
```

Expected: `ERROR` — `_install_dynamic_pixels_hook` not defined.

- [ ] **Step 3: Add `_install_dynamic_pixels_hook` to `pdf_parser_service.py`**

Insert after `_enable_mps_acceleration`:

```python
def _install_dynamic_pixels_hook(ocr_engine) -> None:
    """Wrap vl_rec_model.process to cap min_pixels for OCR blocks.

    All OCR blocks (query prefix "OCR:") get min_pixels capped at
    PDF_VLM_OCR_MAX_PIXELS (default 50176 = 224²).  Table, formula, chart,
    seal, and spotting queries keep their original min_pixels.

    If min_pixels is already <= cap or is None, the call passes through
    unchanged with no overhead.
    """
    outer = ocr_engine.paddlex_pipeline
    pipeline = getattr(outer, "_pipeline", outer)

    if not hasattr(pipeline, "vl_rec_model"):
        return

    vl_model = pipeline.vl_rec_model
    if not hasattr(vl_model, "process"):
        return

    if getattr(vl_model, "_dynamic_pixels_installed", False):
        log.warning("[DynPx] already installed — skipping")
        return

    _orig_process = vl_model.process
    cap = _vlm_ocr_max_pixels()

    def _dynamic_pixels_process(data, min_pixels=None, **kwargs):
        if min_pixels is not None and min_pixels > cap and data:
            query = (data[0].get("query", "") if isinstance(data[0], dict) else "")
            if query.upper().startswith("OCR:"):
                min_pixels = cap
        return _orig_process(data, min_pixels=min_pixels, **kwargs)

    vl_model.process = _dynamic_pixels_process
    vl_model._dynamic_pixels_installed = True
    log.info(
        "[DynPx] OCR min_pixels capped at %d (%dx%d)",
        cap, int(cap ** 0.5), int(cap ** 0.5),
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py::TestDynamicPixelsHook -v 2>&1 | tail -20
```

Expected: 8 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pdf_parser_service.py tests/test_timing_hooks.py
git commit -m "feat: add _install_dynamic_pixels_hook to reduce VLM visual tokens for OCR blocks"
```

---

## Task 4: Wire Up in `run()` and Update Module Docstring

**Files:**
- Modify: `pdf_parser_service.py` — module docstring and `run()`

- [ ] **Step 1: Update the module docstring**

The docstring at the top of `pdf_parser_service.py` lists optional env vars. Add two entries after the `PDF_TIMING` line:

```
  PDF_MPS              Set to "true" to move the VLM model to Apple Metal GPU (MPS).
                       Requires PyTorch with MPS support (Apple Silicon, macOS 13+).
                       Expected speedup: 8-12x for VLM gen step.
  PDF_VLM_OCR_MAX_PIXELS
                       Cap on min_pixels for OCR blocks (default: 50176 = 224x224).
                       Lower values reduce visual token count and speed up prefill.
                       Floor: 12544 (112x112). Does not affect table/formula/chart blocks.
```

- [ ] **Step 2: Update `run()` to call both new functions**

In `run()`, the current hook installation block (around line 938) is:

```python
    if _use_timing():
        _install_timing_hooks(ocr_engine)
        log.info("timing hooks installed (PDF_TIMING=true)")
```

Replace it with:

```python
    if _use_mps():
        _enable_mps_acceleration(ocr_engine)
    _install_dynamic_pixels_hook(ocr_engine)
    if _use_timing():
        _install_timing_hooks(ocr_engine)
        log.info("timing hooks installed (PDF_TIMING=true)")
```

`_install_dynamic_pixels_hook` has no guard because:
- Its default cap (50176) is always less than the pipeline default (112896)
- It always fires for OCR blocks with zero overhead when already at or below cap
- There's no reason to disable it

- [ ] **Step 3: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/test_timing_hooks.py -v 2>&1 | tail -25
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add pdf_parser_service.py
git commit -m "feat: wire MPS acceleration and dynamic pixels hooks into run()"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by |
|---|---|
| `_use_mps()` helper | Task 1 |
| `_vlm_ocr_max_pixels()` helper | Task 1 |
| `_enable_mps_acceleration` — move model to MPS | Task 2 |
| `_enable_mps_acceleration` — patch `_switch_inputs_to_device` | Task 2 |
| `_enable_mps_acceleration` — failure handling | Task 2 (test_mps_to_raises_is_noop, test_mps_not_available_is_noop) |
| `_enable_mps_acceleration` — double-install guard | Task 2 |
| `_install_dynamic_pixels_hook` — OCR cap | Task 3 |
| `_install_dynamic_pixels_hook` — non-OCR pass-through | Task 3 |
| `_install_dynamic_pixels_hook` — double-install guard | Task 3 |
| `run()` call order: MPS → DynPx → Timing | Task 4 |
| Module docstring update | Task 4 |
| TestEnableMpsAcceleration (6 tests) | Task 2 (7 tests, superset) |
| TestDynamicPixelsHook (7 tests) | Task 3 (8 tests, superset) |

All spec requirements covered. ✓

**Placeholder scan:** No TBDs, no "implement later", no "similar to". All steps show exact code. ✓

**Type consistency:**
- `_use_mps()` → `bool` — used as bool guard in Task 4 ✓
- `_vlm_ocr_max_pixels()` → `int` — used as `cap` in `_install_dynamic_pixels_hook` ✓
- `_enable_mps_acceleration(ocr_engine)` — drill-down via `getattr(outer, "_pipeline", outer)` matches existing `_install_timing_hooks` pattern ✓
- `_install_dynamic_pixels_hook(ocr_engine)` — same drill-down ✓
- `vl_model._mps_enabled` flag name — consistent across Task 2 tests and implementation ✓
- `vl_model._dynamic_pixels_installed` flag name — consistent across Task 3 tests and implementation ✓
