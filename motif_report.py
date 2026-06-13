"""Reporting helpers for the motif clustering task."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score



def build_cluster_summary(motifs_df, stat_df, labels, cluster_meta=None):
    df = pd.concat([motifs_df.reset_index(drop=True), stat_df.reset_index(drop=True)], axis=1)
    df["cluster"] = labels.astype(int)
    meta_map = {}
    if cluster_meta is not None and len(cluster_meta):
        meta_map = cluster_meta.set_index("cluster").to_dict("index")

    rows = []
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        meta = meta_map.get(int(cid), {})
        rows.append({
            "cluster": int(cid),
            "count": int(len(sub)),
            "share_pct": round(100 * len(sub) / len(df), 2),
            "length_band": meta.get("length_band", ""),
            "shape_family": meta.get("shape_family", ""),
            "avg_length": round(float(sub["length"].mean()), 2),
            "std_length": round(float(sub["length"].std()), 2),
            "min_length": int(sub["length"].min()),
            "max_length": int(sub["length"].max()),
            "avg_depth": round(float(sub["depth"].mean()), 2),
            "avg_area": round(float(sub["area_below_baseline"].mean()), 2),
            "avg_min_position": round(float(sub["min_position"].mean()), 3),
            "avg_half_width": round(float(sub["half_width_fraction"].mean()), 3),
            "avg_descent_slope": round(float(sub["descent_slope"].mean()), 3),
            "avg_recovery_slope": round(float(sub["recovery_slope"].mean()), 3),
        })
    return pd.DataFrame(rows)


def assign_cluster_names(summary_df):
    """Generate labels from each cluster's rank rather than fragile fixed thresholds."""
    ordered = summary_df.sort_values(["avg_length", "avg_min_position"]).reset_index(drop=True)
    names = {}
    length_words = ["Very short", "Short", "Medium", "Medium", "Long", "Long", "Very long"]
    for rank, row in ordered.iterrows():
        cid = int(row["cluster"])
        length_word = length_words[min(rank, len(length_words) - 1)]
        if row["shape_family"] in ["early-valley", "late-valley"]:
            family = "early valley" if row["avg_min_position"] < 0.5 else "late valley"
            names[cid] = f"{length_word} {family} motif"
        else:
            names[cid] = f"{length_word} compact motif"
    return names


def write_methodology_report(path, cfg, metrics, summary_df, comparison_df, cluster_names, stability, explained_var):
    final_row = comparison_df[comparison_df["method"] == "Final LCSC method"].iloc[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Motif Clustering Methodology Report\n\n")
        f.write("## Executive Summary\n")
        f.write(
            "This version uses a length-constrained shape clustering strategy. The earlier flat k-means approach was cleaner mathematically, "
            "but it mixed very different motif lengths inside the same cluster. The task explicitly states that strongly different lengths should be separated, "
            "so the final method first creates length bands and then performs shape clustering inside the medium and long bands.\n\n"
        )

        f.write("## 1. Motif Extraction\n")
        f.write(f"- Adaptive thresholding: `{cfg.USE_ADAPTIVE_THRESHOLD}`\n")
        f.write(f"- Global threshold suggested by task: `{cfg.GLOBAL_THRESHOLD}` used as reference baseline.\n")
        f.write(f"- Final threshold per signal = percentile({cfg.STEADY_PERCENTILE}) - {cfg.THRESHOLD_OFFSET}\n")
        f.write(f"- Minimum motif length = {cfg.MIN_LEN}\n")
        f.write(f"- Internal gaps up to {cfg.MAX_GAP} samples are merged.\n\n")
        f.write(
            "Reasoning: values around 120 are steady background. A fixed threshold of 110 is a useful first estimate, but a per-signal threshold is safer because "
            "not all columns have exactly the same baseline level.\n\n"
        )

        f.write("## 2. Feature Design\n")
        f.write(f"- Normalized motif curves are resampled to {cfg.FIXED_LEN} points.\n")
        f.write(f"- PCA keeps {cfg.N_PCA_COMPS} components and explains {explained_var * 100:.2f}% of normalized shape variance.\n")
        f.write("- Morphological features include depth, area, valley position, half-width, slopes, skewness, and kurtosis.\n")
        f.write("- Length is represented as log-length to reduce the effect of extreme long motifs.\n\n")

        f.write("## 3. Final Clustering Method: LCSC\n")
        f.write("LCSC = Length-Constrained Shape Clustering.\n\n")
        f.write("1. Cluster motif log-length into 4 ordered length bands.\n")
        f.write("2. Keep very short and short bands as compact length groups.\n")
        f.write("3. Split medium and long bands into early-valley and late-valley shape families.\n")
        f.write("4. Assign stable semantic cluster IDs ordered by length first, then valley position.\n\n")
        f.write("This directly addresses the main review issue: motifs with very different temporal lengths are no longer mixed in the same final cluster.\n\n")

        f.write("## 4. Baseline Comparison\n")
        f.write(comparison_df.round(4).to_markdown(index=False))
        f.write("\n\n")
        f.write(
            "Important interpretation: the final method is selected for task fit, not for maximum full-feature silhouette. The data behaves more like a continuum than cleanly separated islands, "
            "so the strict full-feature silhouette remains modest. However, the task-axis silhouette, length separation, and seed stability are much stronger.\n\n"
        )

        f.write("## 5. Final Evaluation\n")
        f.write(f"- Final number of clusters: {int(final_row['n_clusters'])}\n")
        f.write(f"- Full-feature silhouette score: {float(final_row['full_feature_silhouette']):.4f}\n")
        f.write(f"- Task-axis silhouette score (length + valley timing): {float(final_row['task_axis_silhouette']):.4f}\n")
        f.write(f"- Davies-Bouldin score: {float(final_row['davies_bouldin']):.4f}\n")
        f.write(f"- Calinski-Harabasz score: {float(final_row['calinski_harabasz']):.2f}\n")
        f.write(f"- Length separation score: {float(final_row['length_separation']):.4f}\n")
        f.write(f"- Stability ARI across seeds: {stability[0]:.4f} ± {stability[1]:.4f}\n\n")

        f.write("## 6. Cluster Interpretation\n")
        for _, row in summary_df.iterrows():
            cid = int(row["cluster"])
            f.write(
                f"- Cluster {cid} — {cluster_names[cid]}: n={int(row['count'])}, "
                f"length=[{int(row['min_length'])}, {int(row['max_length'])}], avg_length={row['avg_length']}, "
                f"avg_depth={row['avg_depth']}, valley_position={row['avg_min_position']}\n"
            )
        f.write("\n")

        f.write("## 7. Honest Limitations\n")
        f.write("- Full-feature silhouette is still low, so the motifs should be understood as overlapping families rather than perfectly separated natural classes.\n")
        f.write("- The shape split mainly separates early-valley versus late-valley patterns within length bands. More complex shape types may require DTW or k-shape.\n")
        f.write("- Extraction quality should ideally be validated by manually reviewing 30-50 motifs.\n")
        f.write("- A production version should include a noise/outlier cluster for borderline short segments.\n")
