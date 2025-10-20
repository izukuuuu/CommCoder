"""Standalone demo for generating publication timeline charts."""
from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_output_dir, load_backend


SAMPLE_ROWS = [
    {
        "Publication Year": 2012,
        "研究主题（议题）分类_调整": "教育技术创新",
        "研究领域分类_调整": "教育信息化",
    },
    {
        "Publication Year": 2015,
        "研究主题（议题）分类_调整": "教育技术创新",
        "研究领域分类_调整": "教育信息化",
    },
    {
        "Publication Year": 2017,
        "研究主题（议题）分类_调整": "社区治理",
        "研究领域分类_调整": "社会治理",
    },
    {
        "Publication Year": 2019,
        "研究主题（议题）分类_调整": "社区治理",
        "研究领域分类_调整": "社会治理",
    },
    {
        "Publication Year": 2021,
        "研究主题（议题）分类_调整": "数字政府",
        "研究领域分类_调整": "数字治理",
    },
    {
        "Publication Year": 2023,
        "研究主题（议题）分类_调整": "绿色制造",
        "研究领域分类_调整": "可持续发展",
    },
]


def main() -> None:
    backend = load_backend()
    df = pd.DataFrame(SAMPLE_ROWS)

    timeline = backend._timeline(df, bin_years=4)  # type: ignore[attr-defined]

    output_dir = ensure_output_dir()
    (output_dir / "timeline_data.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2))

    labels = [item["label"] for item in timeline.get("bins", [])]
    counts = [item.get("count", 0) for item in timeline.get("bins", [])]

    fig = backend._render_timeline_line_figure(labels, counts)  # type: ignore[attr-defined]
    (output_dir / "timeline_overview.png").write_bytes(fig_to_png(fig))
    fig.savefig(output_dir / "timeline_overview.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Timeline visualization assets exported to {output_dir}")


def fig_to_png(fig) -> bytes:
    """Serialise a Matplotlib figure to PNG bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    data = buffer.getvalue()
    buffer.close()
    return data


if __name__ == "__main__":
    main()
