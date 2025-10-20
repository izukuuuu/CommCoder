"""Standalone demo for generating keyword word-cloud assets."""
from __future__ import annotations

import json

from utils import decode_data_url, ensure_output_dir, load_backend


SAMPLE_WORDS = [
    {"token": "治理", "count": 120},
    {"token": "数字化", "count": 95},
    {"token": "社区", "count": 88},
    {"token": "协同", "count": 73},
    {"token": "平台", "count": 65},
    {"token": "数据", "count": 60},
    {"token": "学习", "count": 54},
    {"token": "可持续", "count": 48},
    {"token": "创新", "count": 40},
    {"token": "评估", "count": 36},
]


def main() -> None:
    backend = load_backend()
    assets = backend._generate_word_cloud_assets(SAMPLE_WORDS)  # type: ignore[attr-defined]

    output_dir = ensure_output_dir()
    metadata_path = output_dir / "word_cloud_metadata.json"
    metadata = {k: v for k, v in assets.items() if not k.endswith("data_url")}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    png_mime, png_bytes = decode_data_url(assets["png_data_url"])
    svg_mime, svg_bytes = decode_data_url(assets["svg_data_url"])

    (output_dir / "word_cloud.png").write_bytes(png_bytes)
    (output_dir / "word_cloud.svg").write_bytes(svg_bytes)

    print(f"Word cloud assets exported to {output_dir}")


if __name__ == "__main__":
    main()
