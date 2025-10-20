"""Helpers for running standalone visualization demos."""
from __future__ import annotations

import base64
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = PROJECT_ROOT / "重分类gui.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class DatasetContext:
    """Container describing the dataset used for offline visualisation."""

    df: pd.DataFrame
    session_id: str
    source: str
    is_session: bool


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


def load_dataset(
    backend: ModuleType,
    *,
    session_id: Optional[str] = None,
    data_path: Optional[str] = None,
) -> DatasetContext:
    """Load the dataset backing the visualisations.

    Priority order:

    1. Explicit ``data_path`` argument or ``DATA_PATH`` environment variable.
    2. Explicit ``session_id`` argument or ``SESSION_ID`` environment variable.
    3. Most recently updated session found under ``sessions/``.
    4. Latest tabular file inside a ``data`` directory (``*.pkl``/``*.xlsx``/``*.csv``).
    """

    path_candidate = (data_path or os.getenv("DATA_PATH", "")).strip()
    if path_candidate:
        return _load_dataset_from_path(backend, Path(path_candidate))

    session_candidate = (session_id or os.getenv("SESSION_ID", "")).strip()
    if session_candidate:
        return _load_dataset_from_session(backend, session_candidate)

    discovered_sessions = _discover_sessions(backend)
    if discovered_sessions:
        return _load_dataset_from_session(backend, discovered_sessions[0])

    discovered_file = _discover_data_file()
    if discovered_file is not None:
        return _load_dataset_from_path(backend, discovered_file)

    raise RuntimeError(
        "未找到可用的数据集，请使用 --session 或 --data-path 指定，或将数据放在 data/ 目录下。"
    )


def _discover_sessions(backend: ModuleType) -> List[str]:
    try:
        sessions = backend.list_sessions()  # type: ignore[attr-defined]
        ids = [str(entry.get("session_id")) for entry in sessions if entry.get("session_id")]
        if ids:
            return ids
    except Exception:
        pass

    sess_dir = Path(getattr(backend, "SESS_DIR", "sessions"))
    if not sess_dir.is_absolute():
        sess_dir = (BACKEND_PATH.parent / sess_dir).resolve()
    if not sess_dir.exists():
        return []

    candidates: List[Tuple[float, str]] = []
    for child in sess_dir.iterdir():
        if not child.is_dir():
            continue
        data_file = child / "data.pkl"
        if data_file.exists():
            try:
                mtime = data_file.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((mtime, child.name))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in candidates]


def _discover_data_file() -> Optional[Path]:
    search_dirs = [PROJECT_ROOT / "data", PROJECT_ROOT / "Data", PROJECT_ROOT]
    exts = (".pkl", ".xlsx", ".xls", ".csv")
    candidates: List[Tuple[float, Path]] = []
    seen: Dict[Path, float] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for ext in exts:
            for path in directory.glob(f"*{ext}"):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                if path not in seen or seen[path] < mtime:
                    seen[path] = mtime
                    candidates.append((mtime, path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else None


def _load_dataset_from_session(backend: ModuleType, session_id: str) -> DatasetContext:
    df = backend.load_session_from_disk(session_id)  # type: ignore[attr-defined]
    return DatasetContext(
        df=df.copy(),
        session_id=session_id,
        source=f"session:{session_id}",
        is_session=True,
    )


def _load_dataset_from_path(backend: ModuleType, path: Path) -> DatasetContext:
    resolved = path
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"数据文件不存在：{resolved}")

    ext = resolved.suffix.lower()
    if ext == ".pkl":
        df = pd.read_pickle(resolved)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved)
    elif ext == ".csv":
        try:
            df = pd.read_csv(resolved, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(resolved, encoding="gb18030")
    else:
        raise ValueError(f"暂不支持的文件类型：{resolved.suffix}")

    if hasattr(backend, "normalize_columns"):
        df = backend.normalize_columns(df)  # type: ignore[attr-defined]

    slug = re.sub(r"[^0-9A-Za-z]+", "_", resolved.stem).strip("_") or "offline"
    session_id = f"offline_{slug}"
    return DatasetContext(
        df=df.copy(),
        session_id=session_id,
        source=str(resolved),
        is_session=False,
    )
