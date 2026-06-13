"""Visualization functions for the motif clustering task."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from motif_utils import estimate_signal_threshold

COLORS = ["#1B4965", "#2A9D8F", "#F4A261", "#E76F51", "#7B2CBF", "#3A86FF", "#6C757D"]


def _cluster_color_map(labels):
    cluster_ids = sorted(np.unique(labels))
    return {cid: COLORS[i % len(COLORS)] for i, cid in enumerate(cluster_ids)}


def save_method_comparison(comparison_df, output_path):
    metrics = ["task_axis_silhouette", "length_separation", "min_cluster_share"]
    methods = comparison_df["method"].tolist()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(methods))
    width = 0.24
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, comparison_df[m], width, label=m.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_title("Method comparison: quality vs task-fit constraints", fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_cluster_prototypes(resampled, labels, lengths, summary_df, cluster_names, output_path, random_state=42):
    cluster_ids = sorted(np.unique(labels))
    colors = _cluster_color_map(labels)
    ncols = 3
    nrows = int(np.ceil(len(cluster_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)
    rng = np.random.default_rng(random_state)
    x_axis = np.linspace(0, 1, resampled.shape[1])

    for ax, cid in zip(axes, cluster_ids):
        mask = labels == cid
        samples = resampled[mask]
        idx = rng.choice(len(samples), size=min(220, len(samples)), replace=False)
        for curve in samples[idx]:
            ax.plot(x_axis, curve, color="#9CA3AF", alpha=0.09, linewidth=0.7)
        mean_curve = samples.mean(axis=0)
        p10 = np.percentile(samples, 10, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        ax.fill_between(x_axis, p10, p90, color=colors[cid], alpha=0.16)
        ax.plot(x_axis, mean_curve, color=colors[cid], linewidth=3.1)
        row = summary_df[summary_df["cluster"] == cid].iloc[0]
        ax.set_title(
            f"C{cid}: {cluster_names[cid]}\nn={int(row['count'])}, len {int(row['min_length'])}-{int(row['max_length'])}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Normalized time")
        ax.grid(True, alpha=0.18)
        ax.set_ylim(-0.08, 1.08)
    for ax in axes[len(cluster_ids):]:
        ax.axis("off")
    axes[0].set_ylabel("Normalized amplitude")
    fig.suptitle("Final motif clusters: length-constrained prototypes", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_feature_space(features, labels, output_path, random_state=42):
    x_2d = PCA(n_components=2, random_state=random_state).fit_transform(features)
    colors = _cluster_color_map(labels)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1], s=7, alpha=0.32, color=colors[cid], label=f"C{cid} (n={mask.sum()})")
    ax.set_title("2D projection of final feature space", fontweight="bold")
    ax.set_xlabel("PCA projection 1")
    ax.set_ylabel("PCA projection 2")
    ax.legend(markerscale=3, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_length_distribution(lengths, labels, summary_df, output_path):
    colors = _cluster_color_map(labels)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bins = np.linspace(lengths.min(), lengths.max(), 48)
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        ax.hist(lengths[mask], bins=bins, alpha=0.55, color=colors[cid], label=f"C{cid}")
    for _, row in summary_df.iterrows():
        ax.axvline(row["avg_length"], color=colors[int(row["cluster"])], linewidth=1.1, alpha=0.85)
    ax.set_title("Length distribution by final cluster", fontweight="bold")
    ax.set_xlabel("Motif length (samples)")
    ax.set_ylabel("Count")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_cluster_profile(summary_df, output_path):
    cols = ["avg_length", "avg_depth", "avg_area", "avg_min_position", "avg_half_width"]
    profile = summary_df.set_index("cluster")[cols].copy()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(profile_norm.index))
    width = 0.14
    for i, col in enumerate(cols):
        ax.bar(x + (i - 2) * width, profile_norm[col], width, label=col.replace("avg_", "").replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in profile_norm.index])
    ax.set_ylim(0, 1.05)
    ax.set_title("Interpretable cluster profile", fontweight="bold")
    ax.set_ylabel("Min-max normalized value")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _choose_readable_signals(motif_df):
    counts = motif_df.groupby("signal").size()
    readable = counts[(counts >= 3) & (counts <= 7)]
    if len(readable) >= 4:
        return list(readable.sample(4, random_state=42).index)
    return list(counts.sort_values().head(4).index)


def save_sample_signals(df, motif_df, labels, cfg, output_path):
    motif_df = motif_df.copy()
    motif_df["cluster"] = labels.astype(int)
    colors = _cluster_color_map(labels)
    chosen = _choose_readable_signals(motif_df)
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13.5, 10.8), sharex=False)
    if len(chosen) == 1:
        axes = [axes]

    for ax, signal_name in zip(axes, chosen):
        signal = df[signal_name].to_numpy(dtype=float)
        sample_idx = np.arange(len(signal))
        if cfg.USE_ADAPTIVE_THRESHOLD:
            threshold, _ = estimate_signal_threshold(signal, cfg.STEADY_PERCENTILE, cfg.THRESHOLD_OFFSET)
        else:
            threshold = cfg.GLOBAL_THRESHOLD
        ax.plot(signal, sample_idx, color="#111827", linewidth=0.68, alpha=0.88)
        ax.axvline(threshold, color="#C1121F", linestyle="--", linewidth=1.0, alpha=0.75, label=f"threshold={threshold:.1f}")
        sub = motif_df[motif_df["signal"] == signal_name]
        for _, row in sub.iterrows():
            ax.axhspan(row["start"], row["end"], color=colors[int(row["cluster"])], alpha=0.30, label=f"C{int(row['cluster'])}")
        handles, legend_labels = ax.get_legend_handles_labels()
        unique = dict(zip(legend_labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=7, loc="upper right")
        ax.set_title(f"{signal_name}: {len(sub)} detected motifs mapped back to original signal", fontsize=10, fontweight="bold")
        ax.set_xlabel("Signal value")
        ax.set_ylabel("Index")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.18)

    fig.suptitle("Signal-level validation: readable examples, not densest cherry-picked signals", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
