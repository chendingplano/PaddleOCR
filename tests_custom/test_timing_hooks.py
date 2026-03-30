"""Unit tests for timing instrumentation helpers in pdf_parser_service."""
import sys
import os

# Patch third-party modules before importing the service to avoid import errors
sys.modules.setdefault('fitz', __import__('unittest.mock', fromlist=['MagicMock']).MagicMock())
sys.modules.setdefault('paddleocr', __import__('unittest.mock', fromlist=['MagicMock']).MagicMock())
sys.modules.setdefault('psycopg2', __import__('unittest.mock', fromlist=['MagicMock']).MagicMock())
sys.modules.setdefault('numpy', __import__('unittest.mock', fromlist=['MagicMock']).MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pytest
from unittest.mock import MagicMock, patch
from pdf_parser_service import (
    _use_mps, _vlm_ocr_max_pixels, _enable_mps_acceleration, _install_dynamic_pixels_hook, run,
    _use_quantization, _quantize_bits, _enable_quantization,
)


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
        assert "ms" in timing_records[0].message

    def test_logs_ms_only_when_results_are_empty(self, caplog):
        wrapper, _ = self._make_wrapper([], label="EmptyModel")
        with caplog.at_level(logging.INFO):
            list(wrapper())
        timing_records = [r for r in caplog.records if "EmptyModel" in r.message]
        assert len(timing_records) == 1
        assert "boxes=" not in timing_records[0].message
        assert "ms" in timing_records[0].message

    def test_delegates_attribute_access_to_wrapped_model(self):
        from pdf_parser_service import _TimedModelWrapper
        mock_model = MagicMock()
        mock_model.batch_sampler = "sentinel"
        wrapper = _TimedModelWrapper(mock_model, "X")
        assert wrapper.batch_sampler == "sentinel"


# ---------------------------------------------------------------------------
# _install_timing_hooks tests
# ---------------------------------------------------------------------------

class TestInstallTimingHooks:
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
        assert callable(pipeline.get_layout_parsing_results)

    def test_glpr_closure_calls_original_and_returns_result(self):
        from pdf_parser_service import _install_timing_hooks
        engine, pipeline = self._make_mock_engine()
        sentinel = object()
        original_glpr = pipeline.get_layout_parsing_results   # capture BEFORE install
        original_glpr.return_value = sentinel
        _install_timing_hooks(engine)
        result = pipeline.get_layout_parsing_results(
            layout_det_results=[{"boxes": [{"label": "text"}]}],
            images=[],
            imgs_in_doc=[],
        )
        assert result is sentinel
        original_glpr.assert_called_once_with(
            layout_det_results=[{"boxes": [{"label": "text"}]}],
            images=[],
            imgs_in_doc=[],
        )

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

    def test_missing_attributes_are_silently_skipped(self):
        from pdf_parser_service import _install_timing_hooks
        engine, pipeline = self._make_mock_engine(
            has_preprocessor=False, has_layout=False, has_glpr=False
        )
        # Should not raise
        _install_timing_hooks(engine)

    def test_double_install_is_skipped(self, caplog):
        from pdf_parser_service import _install_timing_hooks, _TimedModelWrapper
        engine, pipeline = self._make_mock_engine()
        _install_timing_hooks(engine)
        first_wrapper = pipeline.layout_det_model
        with caplog.at_level(logging.WARNING):
            _install_timing_hooks(engine)   # second call
        assert pipeline.layout_det_model is first_wrapper   # not replaced again
        assert any("already installed" in r.message for r in caplog.records)

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
        # ocr is larger (5022 vs 4022), so ocr first
        assert "ocr" in block_type_records[0].message
        assert "table" in block_type_records[1].message


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


# ---------------------------------------------------------------------------
# _use_mps and _vlm_ocr_max_pixels config helpers tests
# ---------------------------------------------------------------------------

class TestConfigHelpers:
    def test_use_mps_default_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert not _use_mps()

    def test_use_mps_env_true(self):
        with patch.dict(os.environ, {"PDF_MPS": "true"}, clear=True):
            assert _use_mps()

    def test_use_mps_case_insensitive(self):
        with patch.dict(os.environ, {"PDF_MPS": "TRUE"}, clear=True):
            assert _use_mps()

    def test_vlm_ocr_max_pixels_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _vlm_ocr_max_pixels() == 50176

    def test_vlm_ocr_max_pixels_env_override(self):
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "28224"}, clear=True):
            assert _vlm_ocr_max_pixels() == 28224

    def test_vlm_ocr_max_pixels_floor_enforced(self):
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "100"}, clear=True):
            assert _vlm_ocr_max_pixels() == 12544

    def test_vlm_ocr_max_pixels_invalid_returns_default(self):
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "notanumber"}, clear=True):
            assert _vlm_ocr_max_pixels() == 50176


# ---------------------------------------------------------------------------
# _enable_mps_acceleration tests
# ---------------------------------------------------------------------------

def _make_mock_torch(mps_available=True, to_raises=False):
    """Build a mock torch module for _enable_mps_acceleration tests."""
    class FakeTensor:
        def __init__(self, name="t"):
            self.name = name
            self._moved_to = None
        def to(self, device):
            if to_raises:
                raise RuntimeError("unsupported op")
            self._moved_to = device
            return self
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = mps_available
    mock_torch.Tensor = FakeTensor
    return mock_torch, FakeTensor


def _make_vl_engine():
    """Build a minimal fake ocr_engine for _enable_mps_acceleration tests."""
    fake_infer = MagicMock()
    fake_infer.to = lambda device: fake_infer  # .to("mps") returns itself

    fake_vl_model = MagicMock()
    fake_vl_model.infer = fake_infer
    del fake_vl_model._mps_enabled  # ensure attribute absent initially

    fake_pipeline = MagicMock()
    fake_pipeline.vl_rec_model = fake_vl_model

    fake_outer = MagicMock()
    fake_outer._pipeline = fake_pipeline

    fake_engine = MagicMock()
    fake_engine.paddlex_pipeline = fake_outer
    return fake_engine, fake_vl_model, fake_infer


class TestEnableMpsAcceleration:
    def test_mps_moves_model_to_mps(self):
        mock_torch, FakeTensor = _make_mock_torch(mps_available=True)
        engine, vl_model, fake_infer = _make_vl_engine()

        moved = []
        def fake_to(device):
            moved.append(device)
            return fake_infer
        fake_infer.to = fake_to

        with patch.dict(sys.modules, {"torch": mock_torch}):
            _enable_mps_acceleration(engine)

        assert moved == ["mps"]

    def test_mps_patches_switch_inputs(self):
        mock_torch, FakeTensor = _make_mock_torch(mps_available=True)
        engine, vl_model, fake_infer = _make_vl_engine()
        fake_infer.to = lambda device: fake_infer

        with patch.dict(sys.modules, {"torch": mock_torch}):
            _enable_mps_acceleration(engine)

        t = FakeTensor()
        result = vl_model._switch_inputs_to_device({"img": t, "text": "hello"})
        assert result["img"]._moved_to == "mps"
        assert result["text"] == "hello"

    def test_mps_skips_non_tensor_values(self):
        mock_torch, FakeTensor = _make_mock_torch(mps_available=True)
        engine, vl_model, fake_infer = _make_vl_engine()
        fake_infer.to = lambda device: fake_infer

        with patch.dict(sys.modules, {"torch": mock_torch}):
            _enable_mps_acceleration(engine)

        result = vl_model._switch_inputs_to_device({"a": 42, "b": "str", "c": None})
        assert result == {"a": 42, "b": "str", "c": None}

    def test_mps_not_available_is_noop(self, caplog):
        mock_torch, _ = _make_mock_torch(mps_available=False)
        engine, vl_model, fake_infer = _make_vl_engine()
        moved = []
        fake_infer.to = lambda device: moved.append(device) or fake_infer

        import logging
        with caplog.at_level(logging.WARNING):
            with patch.dict(sys.modules, {"torch": mock_torch}):
                _enable_mps_acceleration(engine)

        assert moved == []
        assert not getattr(vl_model, "_mps_enabled", False)

    def test_mps_to_raises_is_noop(self, caplog):
        mock_torch, _ = _make_mock_torch(mps_available=True)
        engine, vl_model, fake_infer = _make_vl_engine()

        def raising_to(device):
            raise RuntimeError("unsupported")
        fake_infer.to = raising_to

        import logging
        with caplog.at_level(logging.WARNING):
            with patch.dict(sys.modules, {"torch": mock_torch}):
                _enable_mps_acceleration(engine)

        assert not getattr(vl_model, "_mps_enabled", False)
        # _switch_inputs_to_device should NOT have been installed as a plain
        # function; it must still be a MagicMock auto-attribute (not replaced).
        assert isinstance(vl_model._switch_inputs_to_device, MagicMock)

    def test_mps_double_install_skipped(self, caplog):
        mock_torch, _ = _make_mock_torch(mps_available=True)
        engine, vl_model, fake_infer = _make_vl_engine()
        fake_infer.to = lambda device: fake_infer

        moved = []
        real_to = fake_infer.to
        def counting_to(device):
            moved.append(device)
            return real_to(device)
        fake_infer.to = counting_to

        import logging
        with caplog.at_level(logging.WARNING):
            with patch.dict(sys.modules, {"torch": mock_torch}):
                _enable_mps_acceleration(engine)
                _enable_mps_acceleration(engine)  # second call

        assert len(moved) == 1  # .to("mps") called only once

    def test_mps_sets_mps_enabled_flag(self):
        mock_torch, _ = _make_mock_torch(mps_available=True)
        engine, vl_model, fake_infer = _make_vl_engine()
        fake_infer.to = lambda device: fake_infer

        with patch.dict(sys.modules, {"torch": mock_torch}):
            _enable_mps_acceleration(engine)

        assert getattr(vl_model, "_mps_enabled", False) is True


# ---------------------------------------------------------------------------
# _install_dynamic_pixels_hook tests
# ---------------------------------------------------------------------------

def _make_pixels_engine():
    """Build a minimal fake engine for _install_dynamic_pixels_hook tests."""
    calls = []

    def fake_process(data, min_pixels=None, **kwargs):
        calls.append({"data": data, "min_pixels": min_pixels, "kwargs": kwargs})
        return "result"

    fake_vl_model = MagicMock()
    fake_vl_model.process = fake_process
    del fake_vl_model._dynamic_pixels_installed

    fake_pipeline = MagicMock()
    fake_pipeline.vl_rec_model = fake_vl_model

    fake_outer = MagicMock()
    fake_outer._pipeline = fake_pipeline

    fake_engine = MagicMock()
    fake_engine.paddlex_pipeline = fake_outer
    return fake_engine, fake_vl_model, calls


class TestDynamicPixelsHook:
    def test_ocr_block_min_pixels_is_capped(self):
        engine, vl_model, calls = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
        data = [{"query": "OCR: please read this text"}]
        vl_model.process(data, min_pixels=112896)
        assert calls[-1]["min_pixels"] == 50176

    def test_non_ocr_block_min_pixels_unchanged(self):
        engine, vl_model, calls = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
        data = [{"query": "Table Recognition: describe the table"}]
        vl_model.process(data, min_pixels=112896)
        assert calls[-1]["min_pixels"] == 112896

    def test_min_pixels_already_below_cap_unchanged(self):
        engine, vl_model, calls = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
        data = [{"query": "OCR: small text"}]
        vl_model.process(data, min_pixels=28224)
        assert calls[-1]["min_pixels"] == 28224

    def test_min_pixels_none_unchanged(self):
        engine, vl_model, calls = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
        data = [{"query": "OCR: text"}]
        vl_model.process(data, min_pixels=None)
        assert calls[-1]["min_pixels"] is None

    def test_formula_block_min_pixels_unchanged(self):
        engine, vl_model, calls = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
        data = [{"query": "Formula Recognition: parse formula"}]
        vl_model.process(data, min_pixels=112896)
        assert calls[-1]["min_pixels"] == 112896

    def test_dynamic_pixels_double_install_skipped(self):
        engine, vl_model, _ = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
            first_process = vl_model.process
            _install_dynamic_pixels_hook(engine)  # second call
        assert vl_model.process is first_process  # not wrapped again

    def test_dynamic_pixels_sets_flag(self):
        engine, vl_model, _ = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
        assert getattr(vl_model, "_dynamic_pixels_installed", False) is True

    def test_ocr_query_case_insensitive(self):
        engine, vl_model, calls = _make_pixels_engine()
        with patch.dict(os.environ, {"PDF_VLM_OCR_MAX_PIXELS": "50176"}, clear=True):
            _install_dynamic_pixels_hook(engine)
        data = [{"query": "ocr: lowercase prefix"}]
        vl_model.process(data, min_pixels=112896)
        assert calls[-1]["min_pixels"] == 50176


# ---------------------------------------------------------------------------
# TestRunHookWiring
# ---------------------------------------------------------------------------

def _run_once_env(**env_overrides):
    """Call run() but break out of the while loop after one iteration."""

    def fake_fetch_candidates(_conn, _batch_size):
        raise SystemExit(0)

    patches = [
        patch("pdf_parser_service._pg_dsn", return_value={"host": "localhost", "dbname": "test", "port": 5432, "user": "u", "password": "p"}),
        patch("pdf_parser_service._repo_dirs", return_value=["/tmp/test_repo"]),
        patch("pdf_parser_service._backup_dir", return_value="/tmp/test_backup"),
        patch("pdf_parser_service._staging_dir", return_value=None),
        patch("pdf_parser_service._use_vl", return_value=True),
        patch("pdf_parser_service._poll_interval", return_value=0),
        patch("pdf_parser_service._batch_size", return_value=1),
        patch("pdf_parser_service._connect", return_value=MagicMock()),
        patch("pdf_parser_service._scan_staging_once"),
        patch("pdf_parser_service._fetch_candidates", side_effect=SystemExit(0)),
        patch("pathlib.Path.mkdir"),
    ]
    return patches



class TestRunHookWiring:
    def test_mps_called_when_pdf_mps_true(self):
        """_enable_mps_acceleration is called when PDF_MPS=true."""
        with patch("pdf_parser_service._enable_mps_acceleration") as mock_mps, \
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
             patch.dict(os.environ, {"PDF_MPS": "true", "PDF_TIMING": "false"}, clear=True):
            mock_ocr_cls.return_value = MagicMock()
            try:
                run()
            except SystemExit:
                pass
        mock_mps.assert_called_once()

    def test_mps_not_called_when_pdf_mps_false(self):
        """_enable_mps_acceleration is NOT called when PDF_MPS is unset."""
        with patch("pdf_parser_service._enable_mps_acceleration") as mock_mps, \
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
             patch.dict(os.environ, {"PDF_TIMING": "false"}, clear=True):
            mock_ocr_cls.return_value = MagicMock()
            try:
                run()
            except SystemExit:
                pass
        mock_mps.assert_not_called()

    def test_dynamic_pixels_always_called(self):
        """_install_dynamic_pixels_hook is always called regardless of env vars."""
        with patch("pdf_parser_service._enable_mps_acceleration"), \
             patch("pdf_parser_service._install_dynamic_pixels_hook") as mock_dyn, \
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
             patch.dict(os.environ, {"PDF_TIMING": "false"}, clear=True):
            mock_ocr_cls.return_value = MagicMock()
            try:
                run()
            except SystemExit:
                pass
        mock_dyn.assert_called_once()

    def test_timing_still_called_when_pdf_timing_true(self):
        """_install_timing_hooks is still called when PDF_TIMING=true."""
        with patch("pdf_parser_service._enable_mps_acceleration"), \
             patch("pdf_parser_service._install_dynamic_pixels_hook"), \
             patch("pdf_parser_service._install_timing_hooks") as mock_timing, \
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
             patch.dict(os.environ, {"PDF_TIMING": "true"}, clear=True):
            mock_ocr_cls.return_value = MagicMock()
            try:
                run()
            except SystemExit:
                pass
        mock_timing.assert_called_once()


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
