# PaddleOCR Operations Guide

Production-ready OCR and document AI engine with support for 100+ languages, including a long-running PDF parser service with Apple Metal GPU (MPS) acceleration.

## Quick Start

```bash
# Navigate to project directory
cd ~/Workspace/ThirdParty/paddleocr/PaddleOCR

# Install PaddlePaddle (CPU/Metal version for Mac)
mise run install-paddle

# Install PaddleOCR with full OCR features (required for VL pipeline)
uv pip install "paddleocr[all]"

# Install service dependencies (torch, PyMuPDF, psycopg2)
mise run install-service-deps

# Run the PDF parser service (CPU)
mise run pdf-parser-service

# Run the PDF parser service with Apple Metal GPU acceleration
mise run pdf-parser-service-mps
```

## Prerequisites

- **Python**: 3.9+ (Python 3.13 in use)
- **uv**: Python package manager
- **mise**: Task runner
- **PostgreSQL**: Required by pdf_parser_service.py (connection configured in mise.local.toml)

## Available Tasks

### Setup

| Task | Description |
|------|-------------|
| `mise run venv` | Create Python virtual environment |
| `mise run update` | Pull latest upstream PaddleOCR changes (`git pull`) |

### Installation

| Task | Description |
|------|-------------|
| `mise run install-paddle` | Install PaddlePaddle framework (CPU version) |
| `mise run install-paddle-gpu` | Install PaddlePaddle framework (GPU/CUDA version) |
| `mise run install-paddleocr` | Install PaddleOCR basic (text recognition only) |
| `mise run install-doc-parser` | Install document parsing features (PP-StructureV3, PaddleOCR-VL) |
| `mise run install-ie` | Install information extraction features (PP-ChatOCRv4) |
| `mise run install-trans` | Install document translation features |
| `mise run install-all` | Install PaddleOCR with all features |
| `mise run install-service-deps` | Install pdf_parser_service.py dependencies (torch, PyMuPDF, psycopg2) |

### PDF Parser Service

| Task | Description |
|------|-------------|
| `mise run pdf-parser-service` | Run PDF parser service (CPU inference) |
| `mise run pdf-parser-service-mps` | Run PDF parser service with Apple Metal GPU (MPS) acceleration |
| `mise run test` | Run unit tests for pdf_parser_service.py |

### Document Processing

| Task | Description | Example |
|------|-------------|---------|
| `mise run ocr [image]` | Run PP-OCRv5 text recognition | `mise run ocr document.png` |
| `mise run structure [image]` | Run PP-StructureV3 document parsing | `mise run structure invoice.png` |
| `mise run doc-parser [image]` | Run PaddleOCR-VL document parsing | `mise run doc-parser receipt.png` |
| `mise run chatocr [image] [key] [api_key]` | Run PP-ChatOCRv4 information extraction | `mise run chatocr id.png 姓名 your_api_key` |
| `mise run parse-pdf [pdf] [output_dir]` | Parse PDF using PaddleOCR | `mise run parse-pdf document.pdf output` |
| `mise run parse-pdf-vl [pdf] [output_dir]` | Parse PDF using PaddleOCR-VL (complex layouts: tables, formulas) | `mise run parse-pdf-vl document.pdf output` |

### Maintenance

| Task | Description |
|------|-------------|
| `mise run verify` | Verify PaddleOCR installation |
| `mise run demo` | Run a quick demo to test installation |
| `mise run clean` | Clean up cached models (`~/.paddleocr`) |
| `mise run help` | Show PaddleOCR CLI help |

## PDF Parser Service

`pdf_parser_service.py` is a long-running service that polls PostgreSQL for pending PDF jobs, parses them using PaddleOCR-VL, and writes results back to the database.

### Environment Variables

Configured in `mise.local.toml` (not committed — contains credentials):

| Variable | Description |
|----------|-------------|
| `PDF_PG_HOST` | PostgreSQL host (default: `127.0.0.1`) |
| `PDF_PG_PORT` | PostgreSQL port (default: `5432`) |
| `PDF_PG_USER` | PostgreSQL user |
| `PDF_PG_PASSWORD` | PostgreSQL password |
| `PDF_PG_DB` | Database name |
| `PDF_REPO_DIRS` | Comma-separated list of directories to scan for PDFs |
| `PDF_BACKUP_DIR` | Directory to move processed PDFs to |
| `PDF_STAGING_DIR` | Staging directory for in-progress work |
| `PDF_POLL_INTERVAL` | Poll interval in seconds (default: `10`) |
| `PDF_BATCH_SIZE` | Max PDFs per batch (default: `25`) |
| `PDF_USE_VL` | Use PaddleOCR-VL pipeline (default: `true`) |

### Performance Tuning Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_MPS` | `false` | Set `true` to move VLM to Apple Metal GPU (~10× speedup on M-series) |
| `PDF_VLM_OCR_MAX_PIXELS` | `50176` | Cap `min_pixels` for OCR blocks (224²=50176). Floor: 12544. Reduces prefill tokens for small blocks. |
| `PDF_TIMING` | `false` | Set `true` to log per-block timing (pre/xfer/gen/post) |

### Expected Performance (M4, 48GB)

| Configuration | Per-page time | Speedup |
|---------------|--------------|---------|
| Baseline (CPU, fixed pixels) | ~58s | 1× |
| `PDF_MPS=true` | ~5.8s | ~10× |
| `PDF_MPS=true` + default `PDF_VLM_OCR_MAX_PIXELS` | ~4.4s | ~13× |

### Initial Setup

```bash
# 1. Install all dependencies
mise run install-paddle
uv pip install "paddlex[ocr]==3.4.3"
mise run install-service-deps

# 2. Configure credentials
cp mise.local.toml.example mise.local.toml   # if example exists, else create manually
# Edit mise.local.toml with your PostgreSQL credentials and paths

# 3. Run with MPS acceleration
mise run pdf-parser-service-mps
```

## Platform-Specific Notes

### macOS (Apple Silicon M1/M2/M3/M4)

```bash
mise run install-paddle          # CPU version (also runs on Metal via PaddlePaddle)
uv pip install "paddlex[ocr]==3.4.3"
mise run install-service-deps    # includes torch for MPS support
```

Enable MPS acceleration for ~10× faster VLM inference:
```bash
PDF_MPS=true uv run --extra service python pdf_parser_service.py
# or
mise run pdf-parser-service-mps
```

**Note:** `PDF_MPS=true` uses PyTorch's Metal backend (MPS) for the VLM forward pass. PaddlePaddle itself still runs on CPU — only the `PaddleOCR-VL` model weights move to the Apple GPU.

### Linux/Windows with NVIDIA GPU

```bash
mise run install-paddle-gpu      # GPU version with CUDA support
mise run install-all
```

## Parsing PDF Documents

### Long-running service (recommended)

For production use, run the PDF parser service which polls a PostgreSQL queue:

```bash
mise run pdf-parser-service-mps
```

### One-off PDF parsing

```bash
# Standard OCR (fast, good for simple documents)
mise run parse-pdf /path/to/document.pdf output/

# VL pipeline (slower, better for tables, formulas, complex layouts)
mise run parse-pdf-vl /path/to/document.pdf output/
```

## Troubleshooting

### `No module named 'fitz'`

```bash
mise run install-service-deps
```

### `No module named 'torch'`

```bash
mise run install-service-deps
```

### `No module named 'paddle'`

```bash
mise run install-paddle
```

### `DependencyError: PaddleOCR-VL-1.5 requires additional dependencies`

The VL pipeline needs `paddlex[ocr]`, not just `paddlex[ocr-core]`:

```bash
uv pip install "paddlex[ocr]==3.4.3"
```

### `No solution found ... safetensors==0.7.0`

This happens when `requires-python = ">=3.8"` is set in `pyproject.toml`. It has been fixed to `>=3.9`. If you see this, pull the latest changes:

```bash
git pull
```

### Model Download Issues

PaddleOCR downloads models on first run to `~/.paddleocr` and `~/.paddlex`. If downloads fail:

- Check internet connection
- Ensure ~2GB free disk space for VL models
- Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip connectivity check (already set in `mise.local.toml`)

### Virtual Environment Issues

```bash
rm -rf .venv
mise run venv
mise run install-paddle
uv pip install "paddlex[ocr]==3.4.3"
mise run install-service-deps
```

## Project Structure

```
PaddleOCR/
├── pdf_parser_service.py     # Long-running PDF parsing service (primary custom file)
├── pdf_parser_service_v1.py  # Previous version (reference)
├── parse_pdf.py              # One-off PDF → OCR utility
├── tests_custom/
│   └── test_timing_hooks.py  # Unit tests for timing/MPS/pixel hooks
├── docs/superpowers/         # Design specs and implementation plans
├── mise.toml                 # Task automation
├── mise.local.toml           # Local credentials/env vars (not committed)
└── pyproject.toml            # Python project config and dependencies
```

## Resources

- **GitHub Repository**: https://github.com/PaddlePaddle/PaddleOCR
- **Documentation**: https://paddlepaddle.github.io/PaddleOCR
- **Design Specs**: `docs/superpowers/specs/`

## License

PaddleOCR is released under the Apache-2.0 License.
