#!/usr/bin/env python3
"""PDF Parser Service

Polls kb.inputs for unprocessed PDF records and runs PaddleOCR on them.
All configuration is read from environment variables.

Required env vars (match the ChenWeb mise.toml [env] section):
  PG_HOST, PG_PORT, PG_DB_NAME, PG_USER_NAME, PG_PASSWORD

Optional env vars:
  STAGING_DIR          Staging directory to scan for new PDF files.
                       Fallback: PDF_STAGING_DIR or DATA_STAGING_DIR
  PDF_REPO_DIR         Base result repo dir. Results are stored under:
                       PDF_REPO_DIR/pdf_parser/<record_id>/
                       Fallback: PDF_REPO_DIRS or DATA_HOME_DIR
  PDF_BACKUP_DIR       Absolute path for PDF backup copies.
                       Default: DATA_BACKUP_DIR/pdf_files or /tmp/pdf-backup/pdf_files
  PDF_POLL_INTERVAL    Seconds between DB polls. Default: 10
  PDF_BATCH_SIZE       Records fetched per poll cycle. Default: 25
  PDF_USE_VL           Set to "true" to use PaddleOCR-VL mode (default: true).
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK
                       Set to "True" to skip connectivity check (speeds up startup).
  PDF_MPS              Set to "true" to move the VLM to Apple Metal GPU (MPS). Default: false.
  PDF_TIMING           Set to "true" to log per-stage timing (layout detection,
                       block count, VLM inference ms) to help profile slowness.
  PDF_VLM_OCR_MAX_PIXELS
                       Cap on min_pixels for OCR blocks (224²=50176). Floor: 12544. Default: 50176.
"""

import fitz  # PyMuPDF — imported early so a missing module fails fast

import json
import logging
import hashlib
import os
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import psycopg2
from paddleocr import PaddleOCR, PaddleOCRVL

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)
log = logging.getLogger(__name__)

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


def _log_block_type_summary(blocks: list) -> None:
    """Emit one [TIMING] BlockType log line per label, sorted by total ms descending."""
    if not blocks:
        return
    totals: dict[str, dict] = {}
    for b in blocks:
        try:
            label = b["label"]
            if label not in totals:
                totals[label] = {"count": 0, "total_ms": 0.0, "total_px": 0, "total_chars": 0}
            t = totals[label]
            t["count"] += 1
            t["total_ms"] += b["pre_ms"] + b["xfer_ms"] + b["gen_ms"] + b["post_ms"]
            t["total_px"] += b["pixels"]
            t["total_chars"] += b["chars"]
        except (KeyError, TypeError):
            pass
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PARSE_OPERATION = "parse"
PARSING_OPERATION = "parsing"
TIME_FORMAT = "%Y%m%d %H:%M:%S"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _repo_dirs() -> list[str]:
    raw = _env("PDF_REPO_DIR") or _env("PDF_REPO_DIRS") or _env("DATA_HOME_DIR")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return ["/tmp/pdf-repo"]


def _backup_dir() -> str:
    custom = _env("PDF_BACKUP_DIR")
    if custom:
        return custom
    base = _env("DATA_BACKUP_DIR") or "/tmp/pdf-backup"
    return os.path.join(base, "pdf_files")


def _staging_dir() -> str:
    return _env("STAGING_DIR") or _env("PDF_STAGING_DIR") or _env("DATA_STAGING_DIR")


def _poll_interval() -> float:
    try:
        return float(_env("PDF_POLL_INTERVAL", "10"))
    except ValueError:
        return 10.0


def _batch_size() -> int:
    try:
        return int(_env("PDF_BATCH_SIZE", "25"))
    except ValueError:
        return 25


def _use_vl() -> bool:
    return _env("PDF_USE_VL", "true").lower() == "true"


def _use_timing() -> bool:
    return _env("PDF_TIMING", "false").lower() == "true"


def _use_mps() -> bool:
    return _env("PDF_MPS", "false").lower() == "true"


def _use_quantization() -> bool:
    return _env("PDF_QUANTIZE_ENABLED", "false").lower() == "true"


def _quantize_bits() -> int:
    """Return 4 or 8. Any value other than '4' is treated as 8."""
    return 4 if _env("PDF_QUANTIZE_BITS", "8").strip() == "4" else 8


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


def _vlm_ocr_max_pixels() -> int:
    """Cap on min_pixels used for OCR blocks. Default 50176 (224²).
    Floor of 12544 (112²) prevents values too small to read text."""
    try:
        return max(12544, int(_env("PDF_VLM_OCR_MAX_PIXELS", "50176")))
    except ValueError:
        return 50176


def _pg_dsn() -> dict:
    return {
        "host": _env("PG_HOST", "127.0.0.1"),
        "port": int(_env("PG_PORT", "5432")),
        "dbname": _env("PG_DB_NAME", "miner"),
        "user": _env("PG_USER_NAME", "admin"),
        "password": _env("PG_PASSWORD"),
    }


# ---------------------------------------------------------------------------
# Status helpers  (mirrors Go status.go logic)
# ---------------------------------------------------------------------------

def _decode_status(raw: str) -> list[dict]:
    raw = (raw or "").strip()
    if not raw or raw == "null":
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def _has_operation(raw_status: str, operation: str) -> bool:
    entries = _decode_status(raw_status)
    op = operation.strip().lower()
    return any(e.get("operation", "").strip().lower() == op for e in entries)


def _append_status(raw_status: str, operation: str, status: str, error: str = "") -> str:
    entries = _decode_status(raw_status)
    entries.append({
        "operation": operation,
        "time": datetime.now().strftime(TIME_FORMAT),
        "status": status,
        "error": error,
    })
    return json.dumps(entries)


def _upsert_status(raw_status: str, operation: str, patch: dict) -> str:
    entries = _decode_status(raw_status)
    op = operation.strip().lower()
    idx = next(
        (i for i, e in enumerate(entries) if e.get("operation", "").strip().lower() == op),
        None,
    )

    merged = {"operation": operation}
    merged.update(patch)

    if idx is None:
        entries.append(merged)
    else:
        current = dict(entries[idx]) if isinstance(entries[idx], dict) else {}
        current.update(merged)
        entries[idx] = current

    return json.dumps(entries)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

_repo_dir_cache: str | None = None

def _choose_repo_dir(repo_dirs: list[str]) -> str:
    """Return the repo dir with the least total bytes.

    Result is cached after the first call — repo dirs rarely change relative
    sizes within a single run, and a full rglob walk on every record is
    prohibitively slow for large repos.
    """
    global _repo_dir_cache
    if _repo_dir_cache is not None and _repo_dir_cache in repo_dirs:
        return _repo_dir_cache

    def _size(d: str) -> int:
        total = 0
        try:
            for f in Path(d).rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except OSError:
            pass
        return total

    best = min(repo_dirs, key=_size)
    Path(best).mkdir(parents=True, exist_ok=True)
    _repo_dir_cache = best
    return best


def _copy_file(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _unique_path(directory: str, filename: str) -> str:
    """Return a path in directory that does not yet exist."""
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = Path(directory) / filename
    i = 1
    while candidate.exists():
        candidate = Path(directory) / f"{base}_{i}{ext}"
        i += 1
    return str(candidate)


def _file_md5(path: str) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


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
            n = sum(len(r.get("boxes", [])) for r in results)
            log.info("[TIMING] %s: %.0fms, boxes=%d", self._label, ms, n)
        else:
            log.info("[TIMING] %s: %.0fms", self._label, ms)
        return results

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def _install_timing_hooks(ocr_engine) -> None:
    """Monkey-patch key pipeline methods to emit [TIMING] log lines.

    Intended to be called only when PDF_TIMING=true.  All patches are applied to the
    live pipeline *instance* — the class and installed paddlex files are
    not modified.

    Log output per page (example):
        [TIMING] PageRender:   38ms
        [TIMING] PNGSave:      22ms
        [TIMING] ArrayBuild:    2ms
        [TIMING] DocPreproc:   12ms
        [TIMING] LayoutDet:   430ms, boxes=14
        [TIMING] Blocks: {'text': 9, 'table': 1, 'formula': 3, 'title': 1}
        [TIMING] Block label=ocr            pixels=142080  pre=   11ms xfer=   1ms gen= 2841ms post=   1ms chars=  183
        [TIMING] Block label=table          pixels=318400  pre=   18ms xfer=   2ms gen= 4120ms post=   2ms chars=  612
        [TIMING] BlockType ocr:             9 blocks  avg= 2950ms  avg_px= 120.0K  avg_chars= 150  total= 26550ms
        [TIMING] BlockType table:           1 block   avg= 4120ms  avg_px= 318.4K  avg_chars= 612  total=  4120ms
        [TIMING] Phases: vlm_loop=30670ms  other=215ms  total=30885ms
        [TIMING] VLM: 14 blocks, 30885ms total, 2206ms avg/block

    Object hierarchy:
        ocr_engine                  PaddleOCRVL
        └─ .paddlex_pipeline        PaddleOCRVL15Pipeline (AutoParallelSimpleInferencePipeline)
               └─ ._pipeline        _PaddleOCRVLPipeline  ← the actual processing object
    The outer pipeline delegates attribute *reads* via __getattr__ to ._pipeline, but
    attribute *writes* land on the outer object.  We must patch ._pipeline directly so
    that self.layout_det_model etc. resolve to our wrappers when the inner pipeline
    executes its predict() method.
    """
    outer = ocr_engine.paddlex_pipeline
    # Drill down to the actual inner pipeline (_PaddleOCRVLPipeline).
    # AutoParallelSimpleInferencePipeline stores it as ._pipeline (single-device mode).
    pipeline = getattr(outer, "_pipeline", outer)
    log.info("timing hooks target: %s", type(pipeline).__name__)

    # Guard against double-install. Check the instance __dict__ to avoid MagicMock auto-creation.
    try:
        if "_timing_hooks_installed" in pipeline.__dict__:
            log.warning("timing hooks already installed — skipping")
            return
    except AttributeError:
        pass
    pipeline._timing_hooks_installed = True

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

        pipeline.get_layout_parsing_results = _timed_glpr

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


def _enable_mps_acceleration(ocr_engine) -> None:
    """Move the VLM PyTorch model to Apple Metal GPU (MPS).

    Patches two things on the live DocVLMPredictor instance:
      1. vl_model.infer        — moved to MPS via .to("mps")
      2. _switch_inputs_to_device — replaced to move torch.Tensor inputs to MPS

    The existing TemporaryDeviceChanger(self.device) context only sets the Paddle
    device and has no effect on PyTorch MPS computation.

    On failure (unsupported op, MPS unavailable), logs a warning and leaves the
    model on CPU — no crash.
    """
    try:
        import torch
    except ModuleNotFoundError:
        log.warning("[MPS] torch not installed — model stays on CPU")
        return

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        log.warning("[MPS] not available — model stays on CPU")
        return

    outer = ocr_engine.paddlex_pipeline
    pipeline = getattr(outer, "_pipeline", outer)

    if not hasattr(pipeline, "vl_rec_model"):
        log.warning("[MPS] vl_rec_model not found — skipping")
        return

    vl_model = pipeline.vl_rec_model

    # Guard against double-install
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


def _install_dynamic_pixels_hook(ocr_engine) -> None:
    """Wrap vl_rec_model.process to cap min_pixels for OCR blocks.

    Only OCR blocks (query prefix "OCR:") are affected. Table, formula, chart,
    seal, and spotting blocks keep their original min_pixels so their complex
    structured output is unaffected.

    If min_pixels is already <= cap (or None), the call is passed through
    unchanged — zero overhead.
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


def _run_ocr(ocr_engine, pdf_path: str, work_dir: str, on_progress) -> tuple[list, int]:
    """Render each page and run OCR using PaddleOCR/PaddleOCR-VL mode.

    Returns (pages, total_pages).  Opens the PDF exactly once.

    Performance notes:
    - Pages are rendered to in-memory numpy arrays (no PNG disk writes).
    - Result JSON is read from the result object's .json property when
      available, falling back to a single reusable temp file only when needed.
    - DB progress updates are throttled to at most one write per 5 pages to
      reduce transaction overhead on long documents.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_ts = time.time()
    on_progress(0, 0)

    tmp_json = os.path.join(work_dir, "_tmp_ocr.json")
    # Throttle: update DB at most once every PROGRESS_STEP pages.
    PROGRESS_STEP = 5

    pages = []

    def _simplify_page_output(page_data: Any, page_number: int, image_filename: str) -> Any:
        """Keep only minimal block fields and include page number per block."""
        if not isinstance(page_data, dict):
            return page_data

        blocks = page_data.get("parsing_res_list")
        if not isinstance(blocks, list):
            return page_data

        simplified_blocks = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            simplified_blocks.append(
                {
                    "block_label": block.get("block_label"),
                    "block_content": block.get("block_content"),
                    "block_bbox": block.get("block_bbox"),
                    "page_number": page_number,
                    "image": image_filename,
                }
            )

        page_data["parsing_res_list"] = simplified_blocks
        return page_data
    try:
        for page_num in range(total_pages):
            page_start_ts = time.time()
            page = doc[page_num]

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

            results = ocr_engine.predict(input=img_array)
            page_data: Any = []
            for res in results:
                # PaddleX result objects expose a .json dict directly.
                # Fall back to the save/reload path only when that's unavailable.
                if hasattr(res, "json") and isinstance(res.json, dict):
                    page_data = res.json
                else:
                    res.save_to_json(tmp_json)
                    with open(tmp_json, "r", encoding="utf-8") as f:
                        page_data = json.load(f)
            pages.append(_simplify_page_output(page_data, page_num + 1, image_filename))

            progress = min(100, int((page_num + 1) * 100 / total_pages)) if total_pages > 0 else 0
            if (page_num + 1) % PROGRESS_STEP == 0 or page_num + 1 == total_pages:
                on_progress(int(time.time() - start_ts), progress)
            print(f"Parsed page {page_num + 1}/{total_pages}, sec:{int(time.time() - page_start_ts)} (MID_26032733)")
    finally:
        doc.close()
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    return pages, total_pages


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------

def _has_md5_record(conn, md5_hex: str) -> bool:
    sql = "SELECT 1 FROM kb.inputs WHERE md5 = %s LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (md5_hex,))
        row = cur.fetchone()
    return row is not None


def _insert_staged_pdf_record(conn, file_path: str, md5_hex: str) -> None:
    status = json.dumps([])
    sql = """
        INSERT INTO kb.inputs (
            name,
            type,
            file_name,
            status,
            md5
        ) VALUES (
            %s,
            'pdf',
            %s,
            %s::jsonb,
            %s
        )
    """
    with conn.cursor() as cur:
        cur.execute(sql, (os.path.basename(file_path), file_path, status, md5_hex))
    conn.commit()


def _move_duplicate_to_backup(src_path: str, backup_dir: str) -> str:
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(src_path))
    if not os.path.exists(backup_path):
        _copy_file(src_path, backup_path)
    if os.path.abspath(src_path) != os.path.abspath(backup_path):
        os.remove(src_path)
    return backup_path


def _scan_staging_once(conn, staging_dir: str, backup_dir: str) -> None:
    if not staging_dir:
        return
    Path(staging_dir).mkdir(parents=True, exist_ok=True)

    for entry in sorted(Path(staging_dir).iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".pdf":
            continue

        src_path = str(entry)
        md5_hex = _file_md5(src_path)

        if _has_md5_record(conn, md5_hex):
            backup_path = _move_duplicate_to_backup(src_path, backup_dir)
            log.info("duplicate staged pdf skipped by md5 and moved to backup: %s -> %s",
                     src_path, backup_path)
            continue

        _insert_staged_pdf_record(conn, src_path, md5_hex)
        log.info("new staged pdf registered: %s md5=%s", src_path, md5_hex)


def _fetch_candidates(conn, batch_size: int) -> list[dict]:
    sql = """
        SELECT id,
               COALESCE(name, ''),
               COALESCE(file_name, ''),
               COALESCE(status::text, '[]')
        FROM kb.inputs
        WHERE LOWER(type) = 'pdf'
          AND (
              status IS NULL
              OR NOT jsonb_path_exists(status, '$[*] ? (@.operation == "parse")')
          )
        ORDER BY id ASC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (batch_size,))
        rows = cur.fetchall()

    return [
        {"id": r[0], "name": r[1], "file_name": r[2], "status": r[3]}
        for r in rows
    ]


def _record_success(conn, rec_id: int, raw_status: str,
                    result_filename: str, file_name: str, backup_filename: str) -> None:
    new_status = _append_status(raw_status, PARSE_OPERATION, "success")
    sql = """
        UPDATE kb.inputs
        SET status          = %s::jsonb,
            result_filename = %s,
            file_name       = %s,
            backup_filename = %s,
            error_msg       = NULL,
            modify_time     = NOW()
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_status, result_filename, file_name, backup_filename, rec_id))
    conn.commit()


def _record_failure(conn, rec_id: int, raw_status: str, error: str) -> None:
    new_status = _append_status(raw_status, PARSE_OPERATION, "fail", error)
    sql = """
        UPDATE kb.inputs
        SET status      = %s::jsonb,
            error_msg   = %s,
            modify_time = NOW()
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_status, error, rec_id))
    conn.commit()


def _record_parsing_active(conn, rec_id: int, raw_status: str,
                           start_time: str, sec_elapsed: int, progress_pct: int) -> str:
    new_status = _upsert_status(
        raw_status,
        PARSING_OPERATION,
        {
            "status": "active",
            "start_time": start_time,
            "sec_elapsed": sec_elapsed,
            "progress": f"{progress_pct}%",
        },
    )
    sql = """
        UPDATE kb.inputs
        SET status      = %s::jsonb,
            modify_time = NOW()
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_status, rec_id))
    conn.commit()
    return new_status


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def _resolve_source_path(file_name: str, staging_dir: str) -> str:
    file_name = (file_name or "").strip()
    if not file_name:
        return ""
    if os.path.isabs(file_name):
        return file_name
    if staging_dir:
        return os.path.join(staging_dir, file_name)
    return file_name


def _print_parse_summary(
    status: str,
    total_pages: int,
    start_time: str,
    total_seconds: int,
    input_file: str,
    result_file: str,
    error: str = "",
) -> None:
    sec_per_page = 0 if total_pages <= 0 else round(total_seconds / total_pages, 3)
    print("Parsing Summary:")
    print(f"Status: {status}")
    print(f"Total Pages: {total_pages}")
    print(f"Start Time: {start_time}")
    print(f"Total Parsing Time (seconds): {total_seconds}")
    print(f"Seconds per Page: {sec_per_page}")
    print(f"Input File: {input_file}")
    print(f"Result File: {result_file}")
    if error:
        print(f"Errors: {error}")


def _process_record(conn, ocr_engine, rec: dict, repo_dirs: list[str],
                    backup_dir: str, staging_dir: str) -> None:
    rec_id = rec["id"]
    source_file = _resolve_source_path(rec["file_name"], staging_dir)
    raw_status = rec["status"]

    if not source_file or not os.path.exists(source_file):
        _record_failure(conn, rec_id, raw_status,
                        f"source file not accessible: {source_file!r}")
        return

    parse_start_dt = datetime.now()
    parse_start = parse_start_dt.strftime(TIME_FORMAT)
    total_pages = 0
    result_path = ""
    try:
        # --- run OCR ---
        print(f"Begin parsing PDF file: input_id={rec_id}, path={source_file}")

        def _progress_update(sec_elapsed: int, progress_pct: int) -> None:
            nonlocal raw_status
            raw_status = _record_parsing_active(
                conn, rec_id, raw_status, parse_start, sec_elapsed, progress_pct
            )

        repo_dir = _choose_repo_dir(repo_dirs)
        record_dir = os.path.join(repo_dir, "pdf_parser", str(rec_id))
        Path(record_dir).mkdir(parents=True, exist_ok=True)

        pages, total_pages = _run_ocr(ocr_engine, source_file, record_dir, _progress_update)

        result = {
            "input_id": rec_id,
            "source_pdf": source_file,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine": "paddleocr",
            "pages": pages,
        }

        # --- write aggregated result JSON ---
        result_path = os.path.join(record_dir, f"ocr_rslt_{rec_id}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # --- copy original PDF to result record directory ---
        repo_pdf_path = os.path.join(record_dir, os.path.basename(source_file))
        _copy_file(source_file, repo_pdf_path)

        # --- backup original PDF ---
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        backup_path = _unique_path(backup_dir, os.path.basename(source_file))
        _copy_file(source_file, backup_path)

        _record_success(conn, rec_id, raw_status,
                        result_path, repo_pdf_path, backup_path)

        if os.path.exists(source_file):
            os.remove(source_file)

        log.info("processed record id=%s file=%s result=%s", rec_id, source_file, result_path)
        total_seconds = int((datetime.now() - parse_start_dt).total_seconds())
        _print_parse_summary("success", total_pages, parse_start, total_seconds, source_file, result_path)

    except Exception as exc:
        _record_failure(conn, rec_id, raw_status, str(exc))
        log.error("failed to process record id=%s: %s", rec_id, exc)
        total_seconds = int((datetime.now() - parse_start_dt).total_seconds())
        _print_parse_summary("fail", total_pages, parse_start, total_seconds, source_file, result_path, str(exc))


# ---------------------------------------------------------------------------
# Main service loop
# ---------------------------------------------------------------------------

def _connect(dsn: dict):
    conn = psycopg2.connect(**dsn)
    conn.autocommit = False
    return conn


def run() -> None:
    dsn = _pg_dsn()
    repo_dirs = _repo_dirs()
    backup_dir = _backup_dir()
    staging_dir = _staging_dir()
    use_vl = _use_vl()
    poll_interval = _poll_interval()
    batch_size = _batch_size()

    log.info("starting pdf_parser_service")
    log.info("  pg_host=%s db=%s", dsn["host"], dsn["dbname"])
    log.info("  repo_dirs=%s", repo_dirs)
    log.info("  backup_dir=%s", backup_dir)
    log.info("  staging_dir=%s", staging_dir if staging_dir else "(not set)")
    log.info("  poll_interval=%ss batch_size=%s use_vl=%s",
             poll_interval, batch_size, use_vl)

    for d in repo_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    Path(backup_dir).mkdir(parents=True, exist_ok=True)

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

    if _use_mps():
        _enable_mps_acceleration(ocr_engine)
    _install_dynamic_pixels_hook(ocr_engine)
    if _use_timing():
        _install_timing_hooks(ocr_engine)
        log.info("timing hooks installed (PDF_TIMING=true)")

    # Graceful shutdown on SIGINT / SIGTERM.
    # sys.exit() is used so that time.sleep() is interrupted immediately
    # (Python 3.5+ resumes sleep after a signal handler that only sets a flag).
    def _stop(_sig, _frame):
        log.info("shutdown signal received: sig=%s frame=%s", _sig, type(_frame).__name__)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print("To enter the loop")
    conn = None
    while True:
        try:
            if conn is None or conn.closed:
                conn = _connect(dsn)
                log.info("connected to postgres")

            _scan_staging_once(conn, staging_dir, backup_dir)
            records = _fetch_candidates(conn, batch_size)
            if not records:
                log.debug("nothing to process")
            else:
                for rec in records:
                    if _has_operation(rec["status"], PARSE_OPERATION):
                        continue
                    _process_record(conn, ocr_engine, rec, repo_dirs, backup_dir, staging_dir)

        except psycopg2.Error as exc:
            log.error("postgres error: %s — reconnecting", exc)
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            conn = None
            time.sleep(poll_interval)
            continue
        except Exception as exc:
            log.error("unexpected error: %s", exc, exc_info=True)

        time.sleep(poll_interval)


if __name__ == "__main__":
    run()
