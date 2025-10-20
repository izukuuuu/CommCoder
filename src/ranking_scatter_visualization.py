"""Standalone demo for generating ranking scatter plot assets."""
from __future__ import annotations

import json

from utils import decode_data_url, ensure_output_dir, load_backend


SAMPLE_POINTS = [
    {"label": "AI 应用", "x": 1.2, "y": 0.4, "count": 34, "cluster": 0},
    {"label": "社区治理", "x": -0.5, "y": 1.1, "count": 28, "cluster": 1},
    {"label": "数字政府", "x": 0.8, "y": -0.9, "count": 26, "cluster": 0},
    {"label": "绿色制造", "x": -1.3, "y": -0.6, "count": 21, "cluster": 2},
    {"label": "教育信息化", "x": 0.4, "y": 0.7, "count": 19, "cluster": 1},
    {"label": "公共健康", "x": -0.9, "y": 1.4, "count": 16, "cluster": 2},
]


def main() -> None:
    backend = load_backend()
    assets = backend._generate_scatter_assets(  # type: ignore[attr-defined]
        SAMPLE_POINTS,
        cluster_count=3,
        cluster_algo="kmeans",
        explained=0.67,
        pipeline_text="TFIDF → PCA",
        total_docs=144,
        slug="ranking_demo",
    )

    output_dir = ensure_output_dir()
    metadata_path = output_dir / "ranking_scatter_metadata.json"
    metadata = {k: v for k, v in assets.items() if not k.endswith("data_url")}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    png_mime, png_bytes = decode_data_url(assets["png_data_url"])
    svg_mime, svg_bytes = decode_data_url(assets["svg_data_url"])

    (output_dir / metadata.get("png_filename", "ranking_scatter.png")).write_bytes(png_bytes)
    (output_dir / metadata.get("svg_filename", "ranking_scatter.svg")).write_bytes(svg_bytes)

    print(f"Ranking scatter assets exported to {output_dir}")


if __name__ == "__main__":
    main()
