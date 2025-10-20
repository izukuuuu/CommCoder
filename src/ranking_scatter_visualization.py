"""Standalone demo for computing and exporting ranking scatter visuals."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from utils import DatasetContext, ensure_output_dir, load_backend, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="生成智能排序散点图（需具备嵌入依赖或 OpenAI 接口）")
    parser.add_argument("--session", help="指定会话 ID，默认选择最近使用的会话")
    parser.add_argument("--data-path", help="直接从文件加载数据 (.pkl/.xlsx/.csv)")
    parser.add_argument(
        "--group-by",
        choices=["topic_orig", "topic_adj"],
        default="topic_adj",
        help="分组依据（与前端一致）",
    )
    parser.add_argument(
        "--algorithm",
        choices=["centroid", "kmeans", "hdbscan"],
        default="centroid",
        help="聚类算法",
    )
    parser.add_argument("--k", type=int, default=3, help="kmeans 聚类数")
    parser.add_argument("--sigma", type=float, default=2.0, help="离群阈值系数 sigma")
    parser.add_argument(
        "--min-cluster-size", type=int, default=5, help="HDBSCAN 最小簇大小（仅在 algorithm=hdbscan 时生效)"
    )
    parser.add_argument(
        "--embedding-source",
        choices=["local", "openai"],
        default="local",
        help="文本嵌入来源（local 需要安装 sentence-transformers）",
    )
    parser.add_argument("--openai-base-url", help="OpenAI API Base URL，可选")
    parser.add_argument("--openai-api-key", help="OpenAI API Key，可选")
    parser.add_argument("--openai-emb-model", help="OpenAI 嵌入模型，可选")
    args = parser.parse_args()

    backend = load_backend()
    dataset: DatasetContext = load_dataset(backend, session_id=args.session, data_path=args.data_path)

    # 将数据放入 DATASTORE，保持与服务端处理逻辑一致。
    backend.DATASTORE = getattr(backend, "DATASTORE", {})  # type: ignore[attr-defined]
    backend.DATASTORE[dataset.session_id] = dataset.df.copy()  # type: ignore[attr-defined]

    fields = {"title": True, "abstract": True, "summary": True}
    local_model = os.getenv("SENTENCE_MODEL", None)
    total, outliers = backend.compute_group_ranking(  # type: ignore[attr-defined]
        df=backend.DATASTORE[dataset.session_id],  # type: ignore[attr-defined]
        group_by=args.group_by,
        fields=fields,
        algorithm=args.algorithm,
        k=max(1, args.k),
        src=args.embedding_source,
        base_url=args.openai_base_url or os.getenv("OPENAI_BASE_URL", ""),
        api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        emb_model=args.openai_emb_model or os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large"),
        local_model_name=local_model,
        sigma=args.sigma,
        min_cluster_size=args.min_cluster_size,
        sid=dataset.session_id,
    )

    scatter: Dict[str, object] = backend.SCATTER_CACHE.get(dataset.session_id)  # type: ignore[attr-defined]
    if not scatter:
        raise RuntimeError("未生成排序散点数据，请检查嵌入依赖或参数设置")

    output_dir = ensure_output_dir()
    (output_dir / "ranking_scatter_raw.json").write_text(json.dumps(scatter, ensure_ascii=False, indent=2))

    figure_path_png = output_dir / "ranking_scatter.png"
    figure_path_svg = output_dir / "ranking_scatter.svg"
    fig = _render_scatter(scatter)
    fig.savefig(figure_path_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_path_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "session": dataset.session_id,
        "source": dataset.source,
        "group_by": args.group_by,
        "algorithm": args.algorithm,
        "k": args.k,
        "sigma": args.sigma,
        "min_cluster_size": args.min_cluster_size,
        "embedding_source": args.embedding_source,
        "total_points": total,
        "outliers": outliers,
        "png_path": str(figure_path_png),
        "svg_path": str(figure_path_svg),
    }
    (output_dir / "ranking_scatter_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(
        "Ranking scatter assets exported to"
        f" {output_dir} using {dataset.source} (points={total}, outliers={outliers})"
    )


def _render_scatter(scatter: Dict[str, object]):
    coords = np.array(scatter.get("coords", []), dtype=float)
    groups = scatter.get("group_labels", [])
    outliers = scatter.get("outliers", [])
    cluster_labels = scatter.get("cluster_labels", [])

    if coords.size == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        ax.text(0.5, 0.5, "暂无散点数据", ha="center", va="center", fontsize=14)
        return fig

    groups_list: List[str] = [str(g) for g in groups]
    unique_groups = sorted(set(groups_list))
    group_to_index = {g: i for i, g in enumerate(unique_groups)}
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for group in unique_groups:
        mask = np.array([grp == group for grp in groups_list])
        if not mask.any():
            continue
        color = cmap(group_to_index[group] % 10)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=f"{group} ({mask.sum()})",
            s=36,
            alpha=0.75,
            color=color,
            edgecolors="white",
            linewidths=0.6,
        )

    outlier_mask = np.array(outliers, dtype=bool)
    if outlier_mask.any():
        ax.scatter(
            coords[outlier_mask, 0],
            coords[outlier_mask, 1],
            s=72,
            facecolors="none",
            edgecolors="#dc2626",
            linewidths=1.2,
            label=f"Outliers ({int(outlier_mask.sum())})",
        )

    if len(set(cluster_labels)) > 1:
        cluster_labels_int = [str(lbl) for lbl in cluster_labels]
        for cluster in sorted(set(cluster_labels_int)):
            mask = np.array([lbl == cluster for lbl in cluster_labels_int])
            if not mask.any():
                continue
            centroid = coords[mask].mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                f"C{cluster}",
                fontsize=9,
                ha="center",
                va="center",
                color="#1f2937",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#94a3b8", lw=0.8),
            )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("智能排序散点分布", fontsize=14)
    ax.grid(True, color="#e5e7eb", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
