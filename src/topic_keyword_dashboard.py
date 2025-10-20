"""Generate an interactive HTML dashboard for topic/field keywords.

This script leverages the backend's existing topic visualization payload so
that keyword statistics stay consistent between the online product and offline
assets. The HTML output renders multiple bar charts using ECharts, each chart
showing the weight of the top keywords for a specific topic or field.

Example usage::

    python -m topic_keyword_dashboard --session SESSION_ID
    python -m topic_keyword_dashboard --data-path data/topic_info.xlsx

The resulting ``topic_keyword_dashboard.html`` file is saved inside the
``src/output`` directory created by :func:`utils.ensure_output_dir`.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List

from utils import DatasetContext, ensure_output_dir, load_backend, load_dataset


def _prepare_category_payload(
    categories: Iterable[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fallback helper that derives keyword data from raw categories."""

    result: List[Dict[str, Any]] = []
    for category in categories:
        words = category.get("top_words") or []
        if not isinstance(words, list):
            continue

        keywords: List[str] = []
        weights: List[float] = []
        counts: List[int] = []
        for word in words:
            token = str(word.get("token", "")).strip()
            if not token:
                continue
            try:
                weight = float(word.get("weight", 0.0) or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            try:
                count = int(word.get("count", 0) or 0)
            except (TypeError, ValueError):
                count = 0
            keywords.append(token)
            weights.append(weight)
            counts.append(count)
            if len(keywords) >= limit:
                break

        if not keywords:
            continue

        label = str(category.get("label", "") or "未分类")
        display_label = str(category.get("display_label", "") or label)
        result.append(
            {
                "topicName": display_label,
                "label": label,
                "keywords": keywords,
                "weights": weights,
                "counts": counts,
                "documentCount": int(category.get("count", 0) or 0),
                "totalTokens": int(category.get("total_tokens", 0) or 0),
            }
        )
    return result


def _prepare_dashboard_entries(
    entries: Iterable[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Normalise keyword dashboard entries coming from the backend payload."""

    result: List[Dict[str, Any]] = []
    for entry in entries:
        keywords = entry.get("keywords") or []
        weights = entry.get("weights") or []
        counts = entry.get("counts") or []
        if not isinstance(keywords, list) or not isinstance(weights, list):
            continue

        limited_keywords = [str(k).strip() for k in keywords if str(k).strip()][:limit]
        if not limited_keywords:
            continue

        limited_weights: List[float] = []
        limited_counts: List[int] = []
        for idx, token in enumerate(limited_keywords):
            weight_raw = weights[idx] if idx < len(weights) else 0.0
            try:
                weight = float(weight_raw or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            limited_weights.append(weight)
            count_raw = counts[idx] if idx < len(counts) else 0
            try:
                count = int(count_raw or 0)
            except (TypeError, ValueError):
                count = 0
            limited_counts.append(count)

        display_label = str(
            entry.get("display_label")
            or entry.get("topicName")
            or entry.get("label")
            or "未分类"
        )

        result.append(
            {
                "topicName": display_label,
                "label": str(entry.get("label", "")),
                "keywords": limited_keywords,
                "weights": limited_weights,
                "counts": limited_counts,
                "documentCount": int(entry.get("document_count", entry.get("documentCount", 0)) or 0),
                "totalTokens": int(entry.get("total_tokens", entry.get("totalTokens", 0)) or 0),
            }
        )
    return result


def _build_html(data: List[Dict[str, Any]], *, title: str) -> str:
    json_payload = json.dumps(data, ensure_ascii=False)
    return f"""
<!DOCTYPE html>
<html lang=\"zh\">
<head>
    <meta charset=\"utf-8\">
    <title>{title}</title>
    <script src=\"https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js\"></script>
    <style>
        body {{
            font-family: \"-apple-system\", \"BlinkMacSystemFont\", \"Segoe UI\", sans-serif;
            margin: 0;
            padding: 24px;
            background: #f8fafc;
        }}
        h1 {{
            text-align: center;
            font-size: 24px;
            margin-bottom: 24px;
            color: #0f172a;
        }}
        .chart-grid {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 16px;
        }}
        .chart-container {{
            flex: 1 1 320px;
            min-width: 280px;
            max-width: 420px;
            height: 320px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
            padding: 12px;
            box-sizing: border-box;
        }}
        @media (max-width: 768px) {{
            body {{ padding: 12px; }}
            .chart-container {{ height: 300px; }}
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id=\"chart_grid\" class=\"chart-grid\"></div>
    <script>
        const keywordData = {json_payload};
        const chartGrid = document.getElementById('chart_grid');
        const colors = [
            '#2563eb', '#16a34a', '#f97316', '#dc2626', '#0891b2',
            '#9333ea', '#64748b', '#ef4444', '#22c55e', '#eab308'
        ];

        keywordData.forEach((item, index) => {{
            const container = document.createElement('div');
            container.className = 'chart-container';
            container.id = `chart_${{index}}`;
            chartGrid.appendChild(container);

            const chart = echarts.init(container);
            chart.setOption({{
                title: {{
                    text: item.topicName,
                    left: 'center',
                    textStyle: {{ fontSize: 14 }}
                }},
                tooltip: {{
                    trigger: 'axis',
                    axisPointer: {{ type: 'shadow' }},
                    formatter: params => {{
                        if (!params.length) return '';
                        const first = params[0];
                        const value = Number(first.value || 0);
                        return `${{first.name}}: ${{(value * 100).toFixed(2)}}%`;
                    }}
                }},
                grid: {{ top: 48, left: 48, right: 24, bottom: 48 }},
                xAxis: {{
                    type: 'category',
                    data: item.keywords,
                    axisLabel: {{ interval: 0, rotate: 30 }}
                }},
                yAxis: {{
                    type: 'value',
                    axisLabel: {{
                        formatter: value => `${{(value * 100).toFixed(0)}}%`
                    }}
                }},
                series: [{{
                    type: 'bar',
                    data: item.weights,
                    itemStyle: {{ color: colors[index % colors.length] }},
                    label: {{
                        show: true,
                        position: 'top',
                        formatter: params => `${{(Number(params.value || 0) * 100).toFixed(1)}}%`
                    }}
                }}]
            }});
        }});
    </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成主题/领域关键词仪表盘 HTML")
    parser.add_argument("--session", help="指定会话 ID，默认选择最近使用的会话")
    parser.add_argument("--data-path", help="直接读取指定数据文件 (.pkl/.xlsx/.csv)")
    parser.add_argument(
        "--dimension",
        choices=["topic", "field"],
        default="topic",
        help="关键词维度：topic 或 field",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="每个分类展示的关键词数量上限 (默认 15)",
    )
    args = parser.parse_args()

    backend = load_backend()
    dataset: DatasetContext = load_dataset(backend, session_id=args.session, data_path=args.data_path)

    if getattr(backend, "SentenceTransformer", None) is not None:
        backend.SentenceTransformer = None  # type: ignore[attr-defined]

    payload: Dict[str, Any] = backend._topic_visual_payload(dataset.df)  # type: ignore[attr-defined]
    dim_map = {"topic": "topic_adj", "field": "field_adj"}
    dim_key = dim_map.get(args.dimension, "topic_adj")
    dim_payload = payload.get(dim_key, {}) if isinstance(payload, dict) else {}

    prepared: List[Dict[str, Any]] = []
    if isinstance(dim_payload, dict):
        raw_dashboard = dim_payload.get("keyword_dashboard")
        if isinstance(raw_dashboard, list) and raw_dashboard:
            prepared = _prepare_dashboard_entries(raw_dashboard, limit=max(args.top, 1))
        if not prepared:
            categories = dim_payload.get("categories")
            if isinstance(categories, list):
                prepared = _prepare_category_payload(categories, limit=max(args.top, 1))

    if not prepared:
        raise RuntimeError("当前数据集中暂无可用于关键词仪表盘的数据")

    title = "研究主题关键词概览" if args.dimension == "topic" else "研究领域关键词概览"
    html_text = _build_html(prepared, title=title)

    output_dir = ensure_output_dir()
    output_path = output_dir / "topic_keyword_dashboard.html"
    output_path.write_text(html_text, encoding="utf-8")

    print(f"关键词仪表盘已生成：{output_path}")


if __name__ == "__main__":
    main()
