"""Standalone demo for generating publication timeline charts."""
from __future__ import annotations

import argparse
import io
import json

import matplotlib.pyplot as plt

from utils import DatasetContext, ensure_output_dir, load_backend, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="导出时间轴折线图数据及图片")
    parser.add_argument("--session", help="指定会话 ID，默认选择最近使用的会话")
    parser.add_argument("--data-path", help="直接从文件加载数据 (.pkl/.xlsx/.csv)")
    parser.add_argument("--bin-years", type=int, default=5, help="时间分组的跨度（年份），默认为 5 年")
    args = parser.parse_args()

    backend = load_backend()
    dataset: DatasetContext = load_dataset(backend, session_id=args.session, data_path=args.data_path)

    timeline = backend._timeline(dataset.df, bin_years=args.bin_years)  # type: ignore[attr-defined]

    output_dir = ensure_output_dir()
    (output_dir / "timeline_data.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2))

    labels = [item["label"] for item in timeline.get("bins", [])]
    counts = [item.get("count", 0) for item in timeline.get("bins", [])]

    fig = backend._render_timeline_line_figure(labels, counts)  # type: ignore[attr-defined]
    (output_dir / "timeline_overview.png").write_bytes(fig_to_png(fig))
    fig.savefig(output_dir / "timeline_overview.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        "Timeline visualization assets exported to"
        f" {output_dir} using {dataset.source} (bin_years={args.bin_years})"
    )


def fig_to_png(fig) -> bytes:
    """Serialise a Matplotlib figure to PNG bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    data = buffer.getvalue()
    buffer.close()
    return data


if __name__ == "__main__":
    main()
