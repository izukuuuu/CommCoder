"""Standalone demo for generating keyword bar chart assets."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from utils import DatasetContext, decode_data_url, ensure_output_dir, load_backend, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="生成关键词条形图图片，基于后端已有数据")
    parser.add_argument("--session", help="指定会话 ID，默认选择最近使用的会话")
    parser.add_argument("--data-path", help="直接从文件加载数据 (.pkl/.xlsx/.csv)")
    parser.add_argument("--dimension", choices=["topic", "field"], default="topic", help="关键词维度：topic 或 field")
    parser.add_argument("--label", help="指定分类名称（默认自动选择数量最多的一类)")
    args = parser.parse_args()

    backend = load_backend()
    if getattr(backend, "SentenceTransformer", None) is not None:
        backend.SentenceTransformer = None  # type: ignore[attr-defined]
    dataset: DatasetContext = load_dataset(backend, session_id=args.session, data_path=args.data_path)

    payload = backend._topic_visual_payload(dataset.df)  # type: ignore[attr-defined]
    dim_map = {"topic": "topic_adj", "field": "field_adj"}
    dim_payload: Dict[str, object] = payload.get(dim_map[args.dimension], {}) if isinstance(payload, dict) else {}
    categories: List[Dict[str, object]] = []
    if isinstance(dim_payload, dict):
        raw = dim_payload.get("categories")
        if isinstance(raw, list):
            categories = [cat for cat in raw if isinstance(cat, dict)]

    if not categories:
        raise RuntimeError("当前数据集中暂无可用于关键词条形图的分类信息")

    target = None
    if args.label:
        for cat in categories:
            if str(cat.get("label", "")) == args.label:
                target = cat
                break
    if target is None:
        target = categories[0]

    words = target.get("top_words") if isinstance(target, dict) else []
    if not isinstance(words, list) or not words:
        raise RuntimeError("目标分类缺少关键词数据，无法生成条形图")

    assets = backend._generate_keyword_bar_assets(words)  # type: ignore[attr-defined]

    output_dir = ensure_output_dir()
    metadata_path = output_dir / "keyword_bar_metadata.json"
    metadata = {k: v for k, v in assets.items() if not k.endswith("data_url")}
    metadata.update(
        {
            "dimension": args.dimension,
            "category": target.get("label") if isinstance(target, dict) else "",
            "source": dataset.source,
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    _, png_bytes = decode_data_url(assets["png_data_url"])
    _, svg_bytes = decode_data_url(assets["svg_data_url"])

    png_name = metadata.get("png_filename", "keyword_bar.png")
    svg_name = metadata.get("svg_filename", "keyword_bar.svg")
    (output_dir / str(png_name)).write_bytes(png_bytes)
    (output_dir / str(svg_name)).write_bytes(svg_bytes)

    print(
        "Keyword bar chart assets exported to"
        f" {output_dir} using {dataset.source} ({args.dimension}:{metadata.get('category','')})"
    )


if __name__ == "__main__":
    main()
