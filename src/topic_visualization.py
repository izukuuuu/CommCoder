"""Standalone demo for generating topic/field scatter visualizations."""
from __future__ import annotations

import json

import pandas as pd

from utils import decode_data_url, ensure_output_dir, load_backend


SAMPLE_ROWS = [
    {
        "Article Title": "AI-Driven Personalised Learning Platforms",
        "Abstract": "We explore artificial intelligence techniques applied to adaptive learning systems in higher education.",
        "结构化总结": "研究高等教育场景中人工智能驱动的个性化学习平台，提出多维度评估框架。",
        "研究主题（议题）分类": "人工智能应用",
        "研究主题（议题）分类_调整": "教育技术创新",
        "研究领域分类": "教育技术",
        "研究领域分类_调整": "教育信息化",
    },
    {
        "Article Title": "Community Resilience After Natural Disasters",
        "Abstract": "Case studies describing community based interventions that accelerated recovery after major flooding events.",
        "结构化总结": "总结洪灾后社区治理与公共服务体系协同的恢复路径。",
        "研究主题（议题）分类": "社会治理",
        "研究主题（议题）分类_调整": "社区治理",
        "研究领域分类": "公共管理",
        "研究领域分类_调整": "社会治理",
    },
    {
        "Article Title": "Data Governance for Smart Cities",
        "Abstract": "The article proposes an evaluation model for data governance maturity in smart city projects across Asia.",
        "结构化总结": "提出智慧城市数据治理成熟度的量化指标体系。",
        "研究主题（议题）分类": "智慧城市",
        "研究主题（议题）分类_调整": "数字政府",
        "研究领域分类": "信息科学",
        "研究领域分类_调整": "数字治理",
    },
    {
        "Article Title": "Sustainable Supply Chains in Manufacturing",
        "Abstract": "Survey of sustainable procurement and emission reporting practices among manufacturing enterprises.",
        "结构化总结": "分析制造业绿色供应链的采购策略与排放核算。",
        "研究主题（议题）分类": "产业升级",
        "研究主题（议题）分类_调整": "绿色制造",
        "研究领域分类": "经济管理",
        "研究领域分类_调整": "可持续发展",
    },
]


def main() -> None:
    backend = load_backend()
    if getattr(backend, "SentenceTransformer", None) is not None:
        # Force TF-IDF fallback to avoid downloading large embedding models.
        backend.SentenceTransformer = None  # type: ignore[attr-defined]
    df = pd.DataFrame(SAMPLE_ROWS)

    payload = backend._topic_visual_payload(df)  # type: ignore[attr-defined]

    output_dir = ensure_output_dir()
    json_path = output_dir / "topic_visuals.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    for dimension, info in payload.items():
        scatter = info.get("scatter", {}) if isinstance(info, dict) else {}
        if not isinstance(scatter, dict):
            continue
        for ext, key in (("png", "png_data_url"), ("svg", "svg_data_url")):
            data_url = scatter.get(key)
            if not data_url:
                continue
            _, binary = decode_data_url(str(data_url))
            filename = scatter.get(f"{ext}_filename") or f"{dimension}_scatter.{ext}"
            (output_dir / filename).write_bytes(binary)

    print(f"Topic visualization data exported to {output_dir}")


if __name__ == "__main__":
    main()
