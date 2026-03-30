# torchao Weight-Only Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PDF_QUANTIZE_ENABLED` / `PDF_QUANTIZE_BITS` env vars that apply torchao INT8 or INT4 weight-only quantization to the VLM at startup, halving or quartering the memory bandwidth cost per forward pass.

**Architecture:** Three new functions in `pdf_parser_service.py` following the existing `_use_mps` / `_enable_mps_acceleration` monkey-patch pattern. Quantization is applied once at startup before MPS (torchao operates on CPU tensors). Controlled entirely by env vars so runs can be compared directly.

**Tech Stack:** Python, torchao 0.16.0, pytest, uv

---

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | Add `torchao` to `service` extra |
| `pdf_parser_service.py` | Add `_use_quantization()`, `_quantize_bits()`, `_enable_quantization()`; update `run()` call order |
| `tests_custom/test_timing_hooks.py` | Add `TestQuantizationHelpers` and `TestEnableQuantization` classes; update import line; extend `TestRunHookWiring` |

---

## Task 1: Add torchao dependency

**Files:**
- Modify: `pyproject.toml:62`

- [ ] **Step 1: Edit pyproject.toml**

Change line 62 from:
```toml
service = ["pymupdf", "torch", "psycopg2-binary"]
```
to:
```toml
service = ["pymupdf", "torch", "psycopg2-binary", "torchao"]
```

- [ ] **Step 2: Verify it resolves**

```bash
uv pip install --extra service --dry-run 2>&1 | grep torchao
```
Expected output contains: `+ torchao==0.16.0`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add torchao to service extra for weight-only quantization"
```

---

## Task 2: Config helpers `_use_quantization` and `_quantize_bits`

**Files:**
- Modify: `pdf_parser_service.py:177` (after `_use_mps`)
- Modify: `tests_custom/test_timing_hooks.py:16` (import line) and append new class

- [ ] **Step 1: Update import line in test file**

Line 16 currently reads:
```python
from pdf_parser_service import _use_mps, _vlm_ocr_max_pixels, _enable_mps_acceleration, _install_dynamic_pixels_hook, run
```
Change to:
```python
from pdf_parser_service import (
    _use_mps, _vlm_ocr_max_pixels, _enable_mps_acceleration, _install_dynamic_pixels_hook, run,
    _use_quantization, _quantize_bits, _enable_quantization,
)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests_custom/test_timing_hooks.py`:
```python
class TestQuantizationHelpers:
    def test_use_quantization_true(self):
        with patch.dict(os.environ, {"PDF_QUANTIZE_ENABLED": "true"}, clear=True):
            assert _use_quantization() is True

    def test_use_quantization_false(self):
        with patch.dict(os.environ, {"PDF_QUANTIZE_ENABLED": "false"}, clear=True):
            assert _use_quantization() is False

    def test_use_quantization_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _use_quantization() is False

    def test_quantize_bits_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _quantize_bits() == 8

    def test_quantize_bits_8(self):
        with patch.dict(os.environ, {"PDF_QUANTIZE_BITS": "8"}, clear=True):
            assert _quantize_bits() == 8

    def test_quantize_bits_4(self):
        with patch.dict(os.environ, {"PDF_QUANTIZE_BITS": "4"}, clear=True):
            assert _quantize_bits() == 4

    def test_quantize_bits_invalid_defaults_to_8(self):
        with patch.dict(os.environ, {"PDF_QUANTIZE_BITS": "16"}, clear=True):
            assert _quantize_bits() == 8
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py::TestQuantizationHelpers -v
```
Expected: `ImportError` or `FAILED` because `_use_quantization` is not yet defined.

- [ ] **Step 4: Implement the helpers**

In `pdf_parser_service.py`, insert after line 177 (`_use_mps` block ends):
```python
def _use_quantization() -> bool:
    return _env("PDF_QUANTIZE_ENABLED", "false").lower() == "true"


def _quantize_bits() -> int:
    """Return 4 or 8. Any value other than '4' is treated as 8."""
    return 4 if _env("PDF_QUANTIZE_BITS", "8").strip() == "4" else 8
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py::TestQuantizationHelpers -v
```
Expected: 7 passed.

- [ ] **Step 6: Run full suite to check no regressions**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py -v
```
Expected: all previously passing tests still pass, plus 7 new.

- [ ] **Step 7: Commit**

```bash
git add pdf_parser_service.py tests_custom/test_timing_hooks.py
git commit -m "feat: add _use_quantization and _quantize_bits config helpers"
```

---

## Task 3: `_enable_quantization` function

**Files:**
- Modify: `pdf_parser_service.py` (after `_enable_mps_acceleration`, around line 611)
- Modify: `tests_custom/test_timing_hooks.py` (append new class)

- [ ] **Step 1: Write the failing tests**

Append to `tests_custom/test_timing_hooks.py`:

```python
def _make_quant_engine():
    """Build a minimal fake ocr_engine for _enable_quantization tests.
    Reuses the same shape as _make_vl_engine."""
    fake_infer = MagicMock()

    fake_vl_model = MagicMock()
    fake_vl_model.infer = fake_infer
    # Ensure _quantization_enabled is absent initially
    del fake_vl_model._quantization_enabled

    fake_pipeline = MagicMock()
    fake_pipeline.vl_rec_model = fake_vl_model

    fake_outer = MagicMock()
    fake_outer._pipeline = fake_pipeline

    fake_engine = MagicMock()
    fake_engine.paddlex_pipeline = fake_outer
    return fake_engine, fake_vl_model, fake_infer


def _make_torchao(int8_instance=None, int4_instance=None, quantize_raises=False):
    """Build a fake torchao.quantization module."""
    mock_quant = MagicMock()
    mock_quant.Int8WeightOnlyConfig.return_value = int8_instance or object()
    mock_quant.Int4WeightOnlyConfig.return_value = int4_instance or object()
    if quantize_raises:
        mock_quant.quantize_.side_effect = RuntimeError("unsupported")
    return mock_quant


class TestEnableQuantization:
    def test_int8_applied(self):
        """quantize_() called with Int8WeightOnlyConfig instance when bits=8."""
        engine, vl_model, fake_infer = _make_quant_engine()
        int8_sentinel = object()
        mock_quant = _make_torchao(int8_instance=int8_sentinel)

        with patch.dict(sys.modules, {"torchao": MagicMock(), "torchao.quantization": mock_quant}), \
             patch.dict(os.environ, {"PDF_QUANTIZE_BITS": "8"}, clear=True):
            _enable_quantization(engine)

        mock_quant.quantize_.assert_called_once_with(fake_infer, int8_sentinel)

    def test_int4_applied(self):
        """quantize_() called with Int4WeightOnlyConfig instance when bits=4."""
        engine, vl_model, fake_infer = _make_quant_engine()
        int4_sentinel = object()
        mock_quant = _make_torchao(int4_instance=int4_sentinel)

        with patch.dict(sys.modules, {"torchao": MagicMock(), "torchao.quantization": mock_quant}), \
             patch.dict(os.environ, {"PDF_QUANTIZE_BITS": "4"}, clear=True):
            _enable_quantization(engine)

        mock_quant.quantize_.assert_called_once_with(fake_infer, int4_sentinel)

    def test_sets_quantization_enabled_flag(self):
        """_quantization_enabled is set on vl_model after success."""
        engine, vl_model, _ = _make_quant_engine()
        mock_quant = _make_torchao()

        with patch.dict(sys.modules, {"torchao": MagicMock(), "torchao.quantization": mock_quant}), \
             patch.dict(os.environ, {}, clear=True):
            _enable_quantization(engine)

        assert vl_model._quantization_enabled is True

    def test_import_error_is_noop(self, caplog):
        """ImportError from torchao → model unchanged, warning logged."""
        engine, vl_model, fake_infer = _make_quant_engine()

        with patch.dict(sys.modules, {"torchao": None, "torchao.quantization": None}), \
             caplog.at_level(logging.WARNING):
            _enable_quantization(engine)

        assert not getattr(vl_model, "_quantization_enabled", False)
        assert "[Quant]" in caplog.text

    def test_quantize_raises_is_noop(self, caplog):
        """quantize_() exception → model stays unchanged, warning logged."""
        engine, vl_model, _ = _make_quant_engine()
        mock_quant = _make_torchao(quantize_raises=True)

        with patch.dict(sys.modules, {"torchao": MagicMock(), "torchao.quantization": mock_quant}), \
             patch.dict(os.environ, {}, clear=True), \
             caplog.at_level(logging.WARNING):
            _enable_quantization(engine)

        assert not getattr(vl_model, "_quantization_enabled", False)
        assert "[Quant]" in caplog.text

    def test_double_install_skipped(self, caplog):
        """Second call is no-op when _quantization_enabled already set."""
        engine, vl_model, fake_infer = _make_quant_engine()
        vl_model._quantization_enabled = True
        mock_quant = _make_torchao()

        with patch.dict(sys.modules, {"torchao": MagicMock(), "torchao.quantization": mock_quant}), \
             patch.dict(os.environ, {}, clear=True), \
             caplog.at_level(logging.WARNING):
            _enable_quantization(engine)

        mock_quant.quantize_.assert_not_called()
        assert "[Quant]" in caplog.text

    def test_no_vl_rec_model_skipped(self, caplog):
        """Missing vl_rec_model → silent skip with warning."""
        fake_pipeline = MagicMock(spec=[])  # no vl_rec_model attribute
        fake_outer = MagicMock()
        fake_outer._pipeline = fake_pipeline
        fake_engine = MagicMock()
        fake_engine.paddlex_pipeline = fake_outer
        mock_quant = _make_torchao()

        with patch.dict(sys.modules, {"torchao": MagicMock(), "torchao.quantization": mock_quant}), \
             patch.dict(os.environ, {}, clear=True), \
             caplog.at_level(logging.WARNING):
            _enable_quantization(fake_engine)

        mock_quant.quantize_.assert_not_called()
        assert "[Quant]" in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py::TestEnableQuantization -v
```
Expected: `FAILED` — `_enable_quantization` not yet defined.

- [ ] **Step 3: Implement `_enable_quantization`**

In `pdf_parser_service.py`, insert after the closing line of `_enable_mps_acceleration` (the `log.info("[MPS]...")` line, around line 610):

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

- [ ] **Step 4: Run new tests**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py::TestEnableQuantization -v
```
Expected: 7 passed.

- [ ] **Step 5: Run full suite**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pdf_parser_service.py tests_custom/test_timing_hooks.py
git commit -m "feat: add _enable_quantization with torchao INT8/INT4 weight-only support"
```

---

## Task 4: Wire into `run()` and add hook-wiring tests

**Files:**
- Modify: `pdf_parser_service.py:1050-1054` (the hook call block in `run()`)
- Modify: `tests_custom/test_timing_hooks.py` (extend `TestRunHookWiring`)

- [ ] **Step 1: Write the failing wiring tests**

Append these methods inside the existing `TestRunHookWiring` class (after the last test method, before the class ends):

```python
    def test_quantization_called_when_enabled(self):
        """_enable_quantization is called when PDF_QUANTIZE_ENABLED=true."""
        with patch("pdf_parser_service._enable_quantization") as mock_quant, \
             patch("pdf_parser_service._enable_mps_acceleration"), \
             patch("pdf_parser_service._install_dynamic_pixels_hook"), \
             patch("pdf_parser_service._install_timing_hooks"), \
             patch("pdf_parser_service.PaddleOCRVL") as mock_ocr_cls, \
             patch("pdf_parser_service._pg_dsn", return_value={"host": "localhost", "dbname": "test", "port": 5432, "user": "u", "password": "p"}), \
             patch("pdf_parser_service._repo_dirs", return_value=["/tmp/test_repo"]), \
             patch("pdf_parser_service._backup_dir", return_value="/tmp/test_backup"), \
             patch("pdf_parser_service._staging_dir", return_value=None), \
             patch("pdf_parser_service._use_vl", return_value=True), \
             patch("pdf_parser_service._poll_interval", return_value=0), \
             patch("pdf_parser_service._batch_size", return_value=1), \
             patch("pdf_parser_service._connect", return_value=MagicMock()), \
             patch("pdf_parser_service._scan_staging_once"), \
             patch("pdf_parser_service._fetch_candidates", return_value=[]), \
             patch("pdf_parser_service.time.sleep", side_effect=SystemExit(0)), \
             patch("pathlib.Path.mkdir"), \
             patch.dict(os.environ, {"PDF_QUANTIZE_ENABLED": "true"}, clear=True):
            mock_ocr_cls.return_value = MagicMock()
            try:
                run()
            except SystemExit:
                pass
        mock_quant.assert_called_once()

    def test_quantization_not_called_when_disabled(self):
        """_enable_quantization is NOT called when PDF_QUANTIZE_ENABLED is unset."""
        with patch("pdf_parser_service._enable_quantization") as mock_quant, \
             patch("pdf_parser_service._enable_mps_acceleration"), \
             patch("pdf_parser_service._install_dynamic_pixels_hook"), \
             patch("pdf_parser_service._install_timing_hooks"), \
             patch("pdf_parser_service.PaddleOCRVL") as mock_ocr_cls, \
             patch("pdf_parser_service._pg_dsn", return_value={"host": "localhost", "dbname": "test", "port": 5432, "user": "u", "password": "p"}), \
             patch("pdf_parser_service._repo_dirs", return_value=["/tmp/test_repo"]), \
             patch("pdf_parser_service._backup_dir", return_value="/tmp/test_backup"), \
             patch("pdf_parser_service._staging_dir", return_value=None), \
             patch("pdf_parser_service._use_vl", return_value=True), \
             patch("pdf_parser_service._poll_interval", return_value=0), \
             patch("pdf_parser_service._batch_size", return_value=1), \
             patch("pdf_parser_service._connect", return_value=MagicMock()), \
             patch("pdf_parser_service._scan_staging_once"), \
             patch("pdf_parser_service._fetch_candidates", return_value=[]), \
             patch("pdf_parser_service.time.sleep", side_effect=SystemExit(0)), \
             patch("pathlib.Path.mkdir"), \
             patch.dict(os.environ, {}, clear=True):
            mock_ocr_cls.return_value = MagicMock()
            try:
                run()
            except SystemExit:
                pass
        mock_quant.assert_not_called()

    def test_quantization_called_before_mps(self):
        """_enable_quantization is called before _enable_mps_acceleration."""
        call_order = []
        with patch("pdf_parser_service._enable_quantization", side_effect=lambda _: call_order.append("quant")), \
             patch("pdf_parser_service._enable_mps_acceleration", side_effect=lambda _: call_order.append("mps")), \
             patch("pdf_parser_service._install_dynamic_pixels_hook"), \
             patch("pdf_parser_service._install_timing_hooks"), \
             patch("pdf_parser_service.PaddleOCRVL") as mock_ocr_cls, \
             patch("pdf_parser_service._pg_dsn", return_value={"host": "localhost", "dbname": "test", "port": 5432, "user": "u", "password": "p"}), \
             patch("pdf_parser_service._repo_dirs", return_value=["/tmp/test_repo"]), \
             patch("pdf_parser_service._backup_dir", return_value="/tmp/test_backup"), \
             patch("pdf_parser_service._staging_dir", return_value=None), \
             patch("pdf_parser_service._use_vl", return_value=True), \
             patch("pdf_parser_service._poll_interval", return_value=0), \
             patch("pdf_parser_service._batch_size", return_value=1), \
             patch("pdf_parser_service._connect", return_value=MagicMock()), \
             patch("pdf_parser_service._scan_staging_once"), \
             patch("pdf_parser_service._fetch_candidates", return_value=[]), \
             patch("pdf_parser_service.time.sleep", side_effect=SystemExit(0)), \
             patch("pathlib.Path.mkdir"), \
             patch.dict(os.environ, {"PDF_QUANTIZE_ENABLED": "true", "PDF_MPS": "true"}, clear=True):
            mock_ocr_cls.return_value = MagicMock()
            try:
                run()
            except SystemExit:
                pass
        assert call_order == ["quant", "mps"], f"Expected quant before mps, got: {call_order}"
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py::TestRunHookWiring::test_quantization_called_when_enabled tests_custom/test_timing_hooks.py::TestRunHookWiring::test_quantization_not_called_when_disabled tests_custom/test_timing_hooks.py::TestRunHookWiring::test_quantization_called_before_mps -v
```
Expected: `FAILED` — `_enable_quantization` not yet called from `run()`.

- [ ] **Step 3: Update `run()` call order**

In `pdf_parser_service.py`, replace the hook block (lines 1050–1054):

```python
    if _use_mps():
        _enable_mps_acceleration(ocr_engine)
    _install_dynamic_pixels_hook(ocr_engine)
    if _use_timing():
        _install_timing_hooks(ocr_engine)
        log.info("timing hooks installed (PDF_TIMING=true)")
```

with:

```python
    if _use_quantization():
        _enable_quantization(ocr_engine)
    if _use_mps():
        _enable_mps_acceleration(ocr_engine)
    _install_dynamic_pixels_hook(ocr_engine)
    if _use_timing():
        _install_timing_hooks(ocr_engine)
        log.info("timing hooks installed (PDF_TIMING=true)")
```

- [ ] **Step 4: Run the new wiring tests**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py::TestRunHookWiring::test_quantization_called_when_enabled tests_custom/test_timing_hooks.py::TestRunHookWiring::test_quantization_not_called_when_disabled tests_custom/test_timing_hooks.py::TestRunHookWiring::test_quantization_called_before_mps -v
```
Expected: 3 passed.

- [ ] **Step 5: Run full suite**

```bash
uv run python -m pytest tests_custom/test_timing_hooks.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pdf_parser_service.py tests_custom/test_timing_hooks.py
git commit -m "feat: wire _enable_quantization into run(), quantize before MPS"
```

---

## Self-Review

**Spec coverage:**
- ✅ `PDF_QUANTIZE_ENABLED` env var → Task 2
- ✅ `PDF_QUANTIZE_BITS=4|8` env var → Task 2
- ✅ `_enable_quantization()` with INT8/INT4 configs → Task 3
- ✅ torchao import failure is a no-op → Task 3 (`test_import_error_is_noop`)
- ✅ `quantize_()` failure is a no-op → Task 3 (`test_quantize_raises_is_noop`)
- ✅ Double-install guard → Task 3 (`test_double_install_skipped`)
- ✅ Call order: quantize before MPS → Task 4 (`test_quantization_called_before_mps`)
- ✅ `pyproject.toml` torchao dependency → Task 1
- ✅ All 10 tests from spec section 7 are present across Tasks 2–3

**Placeholder scan:** None found.

**Type consistency:** `_enable_quantization` is referenced by name in Task 3 (implementation), Task 4 (wiring), and all test patches — consistent throughout.
