"""Generate JSON describing journal distribution by adjusted categories.

This utility mirrors the logic used by the FastAPI backend so that the
statistics stay consistent with the interactive application.  It focuses on
two dimensions: ``研究主题（调整后）`` and ``研究领域（调整后）``.  For each
dimension, the tool groups the dataset by the adjusted category column and the
``Source Title`` column, extracts the top-N journals (default 5) for every
category, and then records an ``others`` bucket representing the remaining
documents.

Example usage::

    python -m journal_distribution --session SESSION_ID --output journals.json

or, alternatively, provide a direct data file::

    python -m journal_distribution --data-path data/latest.pkl --indent 4

The resulting JSON structure is printed to ``stdout`` when ``--output`` is not
specified; otherwise it is written to the requested path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils import DatasetContext, ensure_output_dir, load_backend, load_dataset


def _build_category_entries(
    raw_entries: Iterable[Dict[str, Any]],
    *,
    top_n: int,
) -> List[Dict[str, Any]]:
    """Normalise journal entries to include an ``others`` bucket."""

    result: List[Dict[str, Any]] = []
    for entry in raw_entries:
        items = entry.get("items") or []
        if not isinstance(items, list):
            continue

        limited_items: List[Dict[str, Any]] = []
        for item in items[:top_n]:
            try:
                count = int(item.get("count", 0) or 0)
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            limited_items.append(
                {
                    "label": str(item.get("label", "")),
                    "count": count,
                    "percent": float(item.get("percent", 0.0) or 0.0),
                }
            )

        total = int(entry.get("total", 0) or 0)
        top_count = sum(item["count"] for item in limited_items)
        others = max(total - top_count, 0)
        others_percent = round(others * 100.0 / total, 2) if total else 0.0

        result.append(
            {
                "category": str(entry.get("category", "")),
                "category_display": str(
                    entry.get("category_display") or entry.get("category", "")
                ),
                "total": total,
                "unique_sources": int(entry.get("unique_sources", 0) or 0),
                "top_journals": limited_items,
                "others": {
                    "count": others,
                    "percent": others_percent,
                },
            }
        )

    return result


def _export_json(
    context: DatasetContext,
    backend: Any,
    *,
    top_n: int,
) -> Dict[str, Any]:
    """Generate the JSON payload for both adjusted dimensions."""

    source_col = "Source Title"
    df = context.df

    if source_col not in df.columns:
        raise KeyError(f"数据中缺少 '{source_col}' 列，无法统计期刊分布。")

    topic_col = getattr(backend, "ADJUST_TOPIC_COL", "研究主题（议题）分类_调整")
    field_col = getattr(backend, "ADJUST_FIELD_COL", "研究领域分类_调整")

    top_sources_fn = getattr(backend, "_top_sources_by_category", None)
    if top_sources_fn is None:
        raise RuntimeError("后端缺少 _top_sources_by_category 方法，无法继续。")

    topic_entries = top_sources_fn(
        df,
        topic_col,
        source_col=source_col,
        fallback_category="未分类",
        fallback_source="未提供期刊",
        label_mapping=getattr(backend, "TOPIC_LABEL_MAP", None),
    )
    field_entries = top_sources_fn(
        df,
        field_col,
        source_col=source_col,
        fallback_category="未分类",
        fallback_source="未提供期刊",
        label_mapping=getattr(backend, "FIELD_LABEL_MAP", None),
    )

    return {
        "meta": {
            "source": context.source,
            "session_id": context.session_id,
            "document_count": int(df.shape[0]),
            "top_n": top_n,
            "source_column": source_col,
        },
        "topic": {
            "label": "研究主题（调整后）",
            "column": topic_col,
            "categories": _build_category_entries(topic_entries, top_n=top_n),
        },
        "field": {
            "label": "研究领域（调整后）",
            "column": field_col,
            "categories": _build_category_entries(field_entries, top_n=top_n),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出期刊分布 JSON，包含 Top-N 期刊及 others 统计。"
    )
    parser.add_argument(
        "--session",
        dest="session_id",
        help="使用已存在的 Session ID 加载数据。",
    )
    parser.add_argument(
        "--data-path",
        dest="data_path",
        help="直接指定数据文件（pkl/xlsx/csv）。",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="每个分类保留的期刊数量，默认 5。",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出 JSON 文件路径；若未提供则打印到 stdout。",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON 缩进空格数，默认 2。",
    )

    args = parser.parse_args()

    backend = load_backend()
    context = load_dataset(
        backend,
        session_id=args.session_id,
        data_path=args.data_path,
    )

    payload = _export_json(context, backend, top_n=max(args.top_n, 1))

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            ensure_output_dir()
            output_path = ensure_output_dir() / output_path.name
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=args.indent),
            encoding="utf-8",
        )
        print(f"已写入：{output_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
