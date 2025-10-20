"""Helpers for running standalone visualization demos."""
from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = PROJECT_ROOT / "重分类gui.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def ensure_output_dir() -> Path:
    """Create the output directory if missing and return it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_backend() -> ModuleType:
    """Dynamically load the main backend module for reuse in demos."""
    spec = importlib.util.spec_from_file_location("backend_app", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to locate backend module at {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_data_url(data_url: str) -> Tuple[str, bytes]:
    """Split a data URL and return its mime type and decoded bytes."""
    if not data_url.startswith("data:"):
        raise ValueError("Expected a data URL string")
    header, b64_data = data_url.split(",", 1)
    mime = header.split(";")[0][5:]
    return mime, base64.b64decode(b64_data)
