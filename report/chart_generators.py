"""
Per-check chart generators.

Each function takes check result data and returns a base64-encoded PNG string
that can be embedded directly in the HTML report.

Uses UMAP as the primary dimensionality reduction method for embedding
visualizations, with automatic PCA fallback if umap-learn is not installed.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Try UMAP, fall back to PCA
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _reduce_2d(vectors: np.ndarray, method: str = "auto",
               random_state: int = 42) -> Tuple[np.ndarray, str]:
    """Reduce vectors to 2D using UMAP (preferred) or PCA (fallback).

    Args:
        vectors: (n_samples, n_features) array.
        method: "umap", "pca", or "auto" (tries UMAP first).
        random_state: For reproducibility.

    Returns:
        (projected, method_used) — the 2D coordinates and which method was used.
    """
    if method == "auto":
        method = "umap" if HAS_UMAP else "pca"

    if method == "umap" and HAS_UMAP:
        n_neighbors = min(15, len(vectors) - 1)
        reducer = UMAP(n_components=2, n_neighbors=max(n_neighbors, 2),
                       min_dist=0.1, random_state=random_state)
        projected = reducer.fit_transform(vectors)
        return projected, "UMAP"
    else:
        pca = PCA(n_components=2, random_state=random_state)
        projected = pca.fit_transform(vectors)
        return projected, "PCA"


# ─── Hero: Annotated Embedding Space ──────────────────────────────────

def generate_hero_embedding_plot(
    all_vectors: List[List[float]],
    all_ids: List[str],
    fault_annotations: Optional[Dict[str, List[str]]] = None,
    title: str = "Embedding Space — Fault Overlay"
) -> str:
    """Generate the main UMAP/PCA plot with fault types color-coded.

    Args:
        all_vectors: All vectors from the database.
        all_ids: Corresponding record IDs.
        fault_annotations: Dict mapping fault type name to list of record IDs.
                           e.g. {"zero_vector": ["id1", "id2"], "nan_vector": ["id3"]}
    """
    if len(all_vectors) < 10:
        return _empty_chart("Not enough vectors for embedding visualization")

    matrix = np.array(all_vectors, dtype=float)

    # Replace NaN/Inf with 0 for projection (they'd crash UMAP/PCA)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle dimension mismatches by padding/truncating to the most common dim
    dims = [len(v) for v in all_vectors]
    target_dim = max(set(dims), key=dims.count)
    clean_rows = []
    for i, v in enumerate(all_vectors):
        row = list(v) + [0.0] * max(0, target_dim - len(v))
        clean_rows.append(row[:target_dim])
    matrix = np.nan_to_num(np.array(clean_rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    projected, method = _reduce_2d(matrix)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Build fault lookup
    fault_annotations = fault_annotations or {}
    id_to_fault: Dict[str, str] = {}
    for fault_type, ids in fault_annotations.items():
        for rid in ids:
            id_to_fault[rid] = fault_type

    # Color scheme for fault types
    fault_colors = {
        "zero_vector": "#ef4444",      # red
        "nan_vector": "#f97316",       # orange
        "inf_vector": "#f97316",       # orange
        "wrong_dimension": "#eab308",  # yellow
        "magnitude_outlier": "#a855f7", # purple
        "near_duplicate": "#22c55e",   # green for seeded confirmed duplicate examples
        "similarity_candidate": "#3b82f6",  # blue for unconfirmed vector neighbors
        "missing_metadata": "#06b6d4", # cyan
        "drift_shifted": "#ec4899",    # pink
        "healthy": "#94a3b8",          # slate (default)
    }

    # Separate healthy from faulty
    healthy_mask = np.array([all_ids[i] not in id_to_fault for i in range(len(all_ids))])
    faulty_mask = ~healthy_mask

    # Plot healthy points first (background)
    if healthy_mask.any():
        ax.scatter(projected[healthy_mask, 0], projected[healthy_mask, 1],
                   c=fault_colors["healthy"], alpha=0.3, s=12, label="Healthy", zorder=1)

    # Plot each fault type
    for fault_type, color in fault_colors.items():
        if fault_type == "healthy":
            continue
        fault_ids = fault_annotations.get(fault_type, [])
        if not fault_ids:
            continue
        mask = np.array([all_ids[i] in fault_ids for i in range(len(all_ids))])
        if mask.any():
            ax.scatter(projected[mask, 0], projected[mask, 1],
                       c=color, alpha=0.9, s=40, label=fault_type.replace("_", " ").title(),
                       edgecolors="white", linewidths=0.5, zorder=2)

    ax.set_title(f"{title} ({method})", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method} Dimension 1")
    ax.set_ylabel(f"{method} Dimension 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    return _fig_to_base64(fig)


# ─── Norm Histogram ──────────────────────────────────────────────────

def generate_norm_histogram(
    norms: List[float],
    mean_norm: float,
    std_norm: float,
    z_threshold: float = 3.0,
    outlier_norms: Optional[List[float]] = None
) -> str:
    """Generate a histogram of L2 norms with outlier threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(norms, bins=50, color="#3b82f6", alpha=0.7, edgecolor="white", linewidth=0.5)

    # Mean line
    ax.axvline(mean_norm, color="#22c55e", linewidth=2, linestyle="-",
               label=f"Mean: {mean_norm:.4f}")

    # Threshold lines
    lower = mean_norm - z_threshold * std_norm
    upper = mean_norm + z_threshold * std_norm
    ax.axvline(lower, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"−{z_threshold}σ: {lower:.4f}")
    ax.axvline(upper, color="#ef4444", linewidth=1.5, linestyle="--",
               label=f"+{z_threshold}σ: {upper:.4f}")

    # Mark outliers
    if outlier_norms:
        for n in outlier_norms:
            ax.axvline(n, color="#a855f7", linewidth=1, alpha=0.6)
        ax.scatter(outlier_norms, [0] * len(outlier_norms),
                   c="#a855f7", s=60, zorder=5, marker="^",
                   label=f"Outliers ({len(outlier_norms)})")

    ax.set_title("Vector L2 Norm Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    return _fig_to_base64(fig)


# ─── Dimension Bar Chart ─────────────────────────────────────────────

def generate_dimension_bar_chart(dimensions_found: Dict[int, int]) -> str:
    """Generate a bar chart showing count per vector dimension."""
    fig, ax = plt.subplots(figsize=(8, 4))

    dims = sorted(dimensions_found.keys())
    counts = [dimensions_found[d] for d in dims]
    colors = ["#22c55e" if i == counts.index(max(counts)) else "#ef4444"
              for i in range(len(counts))]

    bars = ax.bar([str(d) for d in dims], counts, color=colors, edgecolor="white")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Vectors per Dimension", fontsize=12, fontweight="bold")
    ax.set_xlabel("Vector Dimension")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()

    return _fig_to_base64(fig)


# ─── Metadata Completeness Chart ─────────────────────────────────────

def generate_metadata_completeness_chart(per_field: Dict[str, Dict]) -> str:
    """Generate a horizontal bar chart showing completeness % per field."""
    fig, ax = plt.subplots(figsize=(10, max(3, len(per_field) * 0.8)))

    fields = list(per_field.keys())
    pcts = [per_field[f].get("completeness_pct", 0) for f in fields]

    colors = ["#22c55e" if p == 100 else "#f59e0b" if p >= 90 else "#ef4444"
              for p in pcts]

    bars = ax.barh(fields, pcts, color=colors, edgecolor="white", height=0.6)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.axvline(100, color="#22c55e", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Metadata Field Completeness", fontsize=12, fontweight="bold")
    ax.set_xlabel("Completeness %")
    ax.grid(True, alpha=0.15, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()

    return _fig_to_base64(fig)


# ─── Drift Comparison Plot ───────────────────────────────────────────

def generate_drift_comparison_plot(
    reference_vectors: List[List[float]],
    current_vectors: List[List[float]],
    frechet_distance: Optional[float] = None
) -> str:
    """Generate a side-by-side UMAP/PCA plot of reference vs current vectors."""
    if len(reference_vectors) < 10 or len(current_vectors) < 10:
        return _empty_chart("Not enough vectors for drift visualization")

    ref = np.array(reference_vectors, dtype=float)
    cur = np.array(current_vectors, dtype=float)

    if ref.shape[1] != cur.shape[1]:
        return _empty_chart(f"Dimension mismatch: ref={ref.shape[1]}, cur={cur.shape[1]}")

    combined = np.vstack([ref, cur])
    projected, method = _reduce_2d(combined)

    ref_proj = projected[:len(ref)]
    cur_proj = projected[len(ref):]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(ref_proj[:, 0], ref_proj[:, 1], alpha=0.4, s=12,
               c="#3b82f6", label=f"Reference (n={len(ref)})")
    ax.scatter(cur_proj[:, 0], cur_proj[:, 1], alpha=0.4, s=12,
               c="#ef4444", label=f"Current (n={len(cur)})")

    # Draw centroids
    ref_centroid = ref_proj.mean(axis=0)
    cur_centroid = cur_proj.mean(axis=0)
    ax.scatter(*ref_centroid, c="#1e40af", s=200, marker="X", zorder=5,
               edgecolors="white", linewidths=2, label="Reference Centroid")
    ax.scatter(*cur_centroid, c="#991b1b", s=200, marker="X", zorder=5,
               edgecolors="white", linewidths=2, label="Current Centroid")

    # Arrow from ref centroid to cur centroid
    ax.annotate("", xy=cur_centroid, xytext=ref_centroid,
                arrowprops=dict(arrowstyle="->", color="#334155", lw=2))

    title = f"Drift Comparison ({method})"
    if frechet_distance is not None:
        title += f" — Fréchet Distance: {frechet_distance:.4f}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{method} Dimension 1")
    ax.set_ylabel(f"{method} Dimension 2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    return _fig_to_base64(fig)


# ─── Duplicate Network Plot ──────────────────────────────────────────

def generate_duplicate_network_plot(
    all_vectors: List[List[float]],
    all_ids: List[str],
    duplicate_pairs: List[Dict]
) -> str:
    """Generate a UMAP/PCA plot connecting high-similarity candidate pairs."""
    if len(all_vectors) < 10 or not duplicate_pairs:
        return _empty_chart("No similarity candidates to visualize")

    matrix = np.nan_to_num(np.array(all_vectors, dtype=float), nan=0.0)

    # Handle dimension mismatches
    dims = [len(v) for v in all_vectors]
    target_dim = max(set(dims), key=dims.count)
    clean_rows = [list(v)[:target_dim] + [0.0] * max(0, target_dim - len(v))
                  for v in all_vectors]
    matrix = np.nan_to_num(np.array(clean_rows, dtype=float), nan=0.0)

    projected, method = _reduce_2d(matrix)

    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all points
    ax.scatter(projected[:, 0], projected[:, 1], c="#94a3b8", alpha=0.3, s=10)

    # Highlight and connect candidate pairs
    dup_ids = set()
    for pair in duplicate_pairs:
        id_a = pair.get("id_a", "")
        id_b = pair.get("id_b", "")
        if id_a in id_to_idx and id_b in id_to_idx:
            idx_a = id_to_idx[id_a]
            idx_b = id_to_idx[id_b]
            ax.plot([projected[idx_a, 0], projected[idx_b, 0]],
                    [projected[idx_a, 1], projected[idx_b, 1]],
                    c="#3b82f6", linewidth=1.5, alpha=0.7)
            dup_ids.add(id_a)
            dup_ids.add(id_b)

    # Highlight candidate points
    dup_mask = np.array([all_ids[i] in dup_ids for i in range(len(all_ids))])
    if dup_mask.any():
        ax.scatter(projected[dup_mask, 0], projected[dup_mask, 1],
                   c="#3b82f6", s=50, alpha=0.9, edgecolors="white",
                   linewidths=0.5, label=f"Candidates ({len(dup_ids)} records)",
                   zorder=3)

    ax.set_title(f"High-Similarity Candidates ({method})", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{method} Dimension 1")
    ax.set_ylabel(f"{method} Dimension 2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    return _fig_to_base64(fig)


# ─── Helper ──────────────────────────────────────────────────────────

def _empty_chart(message: str) -> str:
    """Generate a placeholder chart with a message."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=12, color="#94a3b8", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    return _fig_to_base64(fig)
