"""Standalone demo for generating topic/field scatter visualizations."""
from __future__ import annotations
import argparse
import json

from utils import DatasetContext, decode_data_url, ensure_output_dir, load_backend, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate topic/field scatter assets from backend data")
    parser.add_argument("--session", help="指定会话 ID，默认选择最近使用的会话")
    parser.add_argument("--data-path", help="直接读取指定数据文件 (.pkl/.xlsx/.csv)")
    args = parser.parse_args()

    backend = load_backend()
    dataset: DatasetContext = load_dataset(backend, session_id=args.session, data_path=args.data_path)

    if getattr(backend, "SentenceTransformer", None) is not None:
        # Avoid downloading large embedding models when running offline scripts.
        backend.SentenceTransformer = None  # type: ignore[attr-defined]

    payload = backend._topic_visual_payload(dataset.df)  # type: ignore[attr-defined]

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

    print(f"Topic visualization data exported to {output_dir} using {dataset.source}")


if __name__ == "__main__":
    main()
