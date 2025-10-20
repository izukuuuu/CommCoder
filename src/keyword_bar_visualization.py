"""Standalone demo for generating keyword bar chart assets."""
from __future__ import annotations

import json

from utils import decode_data_url, ensure_output_dir, load_backend


SAMPLE_WORDS = [
    {"token": "治理", "count": 120, "weight": 0.18},
    {"token": "数字化", "count": 95, "weight": 0.14},
    {"token": "社区", "count": 88, "weight": 0.12},
    {"token": "平台", "count": 74, "weight": 0.11},
    {"token": "协同", "count": 66, "weight": 0.09},
    {"token": "评估", "count": 52, "weight": 0.07},
    {"token": "创新", "count": 45, "weight": 0.06},
]


def main() -> None:
    backend = load_backend()
    assets = backend._generate_keyword_bar_assets(SAMPLE_WORDS)  # type: ignore[attr-defined]

    output_dir = ensure_output_dir()
    metadata_path = output_dir / "keyword_bar_metadata.json"
    metadata = {k: v for k, v in assets.items() if not k.endswith("data_url")}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    png_mime, png_bytes = decode_data_url(assets["png_data_url"])
    svg_mime, svg_bytes = decode_data_url(assets["svg_data_url"])

    (output_dir / metadata.get("png_filename", "keyword_bar.png")).write_bytes(png_bytes)
    (output_dir / metadata.get("svg_filename", "keyword_bar.svg")).write_bytes(svg_bytes)

    print(f"Keyword bar chart assets exported to {output_dir}")


if __name__ == "__main__":
    main()
