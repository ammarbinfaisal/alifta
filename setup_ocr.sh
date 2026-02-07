#!/bin/bash
# Setup script for GLM-OCR (uv + .venv)

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://github.com/astral-sh/uv)."
  exit 1
fi

echo "Creating/using virtual environment '.venv'..."
uv venv .venv

echo "Installing dependencies into '.venv'..."
uv pip install -p .venv \
  torch torchvision torchaudio \
  transformers accelerate pillow pymupdf

echo
echo "Setup complete!"
echo "Run with:"
echo "  .venv/bin/python ocr_mdx_pipeline.py <pdf_path> <vol_num>"
