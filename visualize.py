"""
Embedding space visualization.

Produces a 2D scatter plot of vectors using PCA dimensionality reduction.
This is a diagnostic tool — it helps you visually spot clusters, outliers,
and drift between different batches of embeddings.

No model is trained here — PCA is a linear projection, not a learned model.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt

from connectors.base import VectorDBConnector


def plot_embedding_space(connector: VectorDBConnector,
                          sample_size: int = 2000,
                          output_path: str = "embedding_space.png",
                          title: str = "Embedding Space (PCA 2D Projection)",
                          color_by_field: Optional[str] = None) -> Dict:
    """Sample vectors from the database and plot them in 2D using PCA.

    Args:
        sample_size: How many vectors to sample.
        output_path: Where to save the PNG file.
        title: Plot title.
        color_by_field: If provided, color points by this metadata field's
                        value. Useful for seeing whether different document
                        types cluster separately.

    Returns:
        A result dict with the output path and basic statistics.
    """
    records = []
    vectors = []
    for record in connector.scan(batch_size=500):
        if record.vector:
            records.append(record)
            vectors.append(record.vector)
        if len(vectors) >= sample_size:
            break

    if len(vectors) < 10:
        return {
            "check": "embedding_visualization",
            "passed": None,
            "detail": "Not enough vectors to produce a meaningful plot.",
        }

    matrix = np.array(vectors, dtype=float)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(12, 8))

    if color_by_field:
        # Color by metadata field value
        labels = [r.metadata.get(color_by_field, "unknown") for r in records]
        unique_labels = list(set(labels))
        color_map = {label: i for i, label in enumerate(unique_labels)}
        colors = [color_map[l] for l in labels]
        scatter = ax.scatter(projected[:, 0], projected[:, 1],
                             c=colors, cmap="tab20", alpha=0.5, s=8)
        # Add legend for up to 20 categories
        if len(unique_labels) <= 20:
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=plt.cm.tab20(color_map[l] / max(len(unique_labels), 1)),
                                   markersize=8, label=str(l)[:30])
                       for l in unique_labels]
            ax.legend(handles=handles, loc="best", fontsize=7)
    else:
        ax.scatter(projected[:, 0], projected[:, 1], alpha=0.4, s=8, c="steelblue")

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "check": "embedding_visualization",
        "passed": None,  # informational
        "detail": {
            "output_path": output_path,
            "vectors_plotted": len(vectors),
            "pca_variance_explained": {
                "PC1": round(float(pca.explained_variance_ratio_[0]) * 100, 2),
                "PC2": round(float(pca.explained_variance_ratio_[1]) * 100, 2),
                "total": round(float(sum(pca.explained_variance_ratio_)) * 100, 2),
            },
        },
    }


def plot_drift_comparison(reference_vectors: List[List[float]],
                           current_vectors: List[List[float]],
                           output_path: str = "drift_comparison.png",
                           title: str = "Drift Comparison (PCA 2D)") -> Dict:
    """Plot reference and current vectors side by side in the same PCA
    space to visually show drift.

    Args:
        reference_vectors: Vectors from the reference (healthy) period.
        current_vectors: Vectors from the current period.
        output_path: Where to save the PNG file.
        title: Plot title.
    """
    if len(reference_vectors) < 10 or len(current_vectors) < 10:
        return {
            "check": "drift_visualization",
            "passed": None,
            "detail": "Not enough vectors in one or both snapshots for a meaningful plot.",
        }

    ref = np.array(reference_vectors, dtype=float)
    cur = np.array(current_vectors, dtype=float)

    if ref.shape[1] != cur.shape[1]:
        return {
            "check": "drift_visualization",
            "passed": None,
            "detail": f"Dimension mismatch: reference={ref.shape[1]}, current={cur.shape[1]}.",
        }

    # Fit PCA on combined data so both sets are in the same projection
    combined = np.vstack([ref, cur])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    ref_proj = projected[:len(ref)]
    cur_proj = projected[len(ref):]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(ref_proj[:, 0], ref_proj[:, 1], alpha=0.4, s=8,
               c="steelblue", label=f"Reference (n={len(ref)})")
    ax.scatter(cur_proj[:, 0], cur_proj[:, 1], alpha=0.4, s=8,
               c="coral", label=f"Current (n={len(cur)})")

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "check": "drift_visualization",
        "passed": None,
        "detail": {
            "output_path": output_path,
            "reference_count": len(ref),
            "current_count": len(cur),
            "pca_variance_explained": {
                "PC1": round(float(pca.explained_variance_ratio_[0]) * 100, 2),
                "PC2": round(float(pca.explained_variance_ratio_[1]) * 100, 2),
            },
        },
    }
