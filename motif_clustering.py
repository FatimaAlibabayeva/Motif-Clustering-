"""
Motif Clustering in Time Series Data
====================================
Author: Fatima Alibabayeva

Final approach: Length-Constrained Shape Clustering (LCSC)
1) Extract candidate motifs with adaptive thresholding.
2) Represent each motif by normalized shape, morphology, and length.
3) First separate motifs into length bands, because the task explicitly says
   that strongly different motif lengths should not be grouped together.
4) Within medium/long length bands, split by shape/valley-position family.
5) Export results, metrics, figures, and a methodology report.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import warnings
 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import motif_config as cfg
from motif_plots import (
    save_method_comparison,
    save_cluster_prototypes,
    save_length_distribution,
    save_feature_space,
    save_cluster_profile,
    save_sample_signals,
)
from motif_report import (
    assign_cluster_names,
    build_cluster_summary,
    write_methodology_report,
)
from motif_utils import extract_all_motifs, length_separation_score

warnings.filterwarnings("ignore")

STAT_COLS = [
    "depth", "amplitude", "area_below_baseline", "min_position",
    "half_width_fraction", "descent_slope", "recovery_slope",
    "start_end_delta", "std_value", "skewness", "kurtosis",
]
SHAPE_SPLIT_COLS = [
    "min_position", "half_width_fraction", "depth", "area_below_baseline",
    "descent_slope", "recovery_slope",
]


def build_feature_blocks(resampled_shapes, stat_df):
    """Return standardized feature groups and PCA explained variance."""
    pca = PCA(n_components=cfg.N_PCA_COMPS, random_state=cfg.RANDOM_STATE)
    shape_pca = pca.fit_transform(resampled_shapes)
    shape_block = StandardScaler().fit_transform(shape_pca)
    stats_block = StandardScaler().fit_transform(stat_df[STAT_COLS].to_numpy())
    length_block = StandardScaler().fit_transform(stat_df[["log_length"]].to_numpy())
    return shape_block, stats_block, length_block, float(pca.explained_variance_ratio_.sum())


def task_feature_space(shape_block, stats_block, length_block):
    """Feature space used for metrics and visualisation."""
    return np.hstack([
        shape_block * 1.1,
        stats_block * 1.0,
        length_block * 1.4,
    ])


def length_only_baseline(length_block):
    labels = KMeans(
        n_clusters=cfg.N_LENGTH_BANDS,
        random_state=cfg.RANDOM_STATE,
        n_init=30,
    ).fit_predict(length_block)
    order = sorted(np.unique(labels), key=lambda c: float(length_block[labels == c].mean()))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels])


def flat_kmeans_baseline(features):
    """A simple baseline: direct k-means on all features."""
    return MiniBatchKMeans(
        n_clusters=6,
        random_state=cfg.RANDOM_STATE,
        n_init=20,
        batch_size=cfg.BATCH_SIZE,
    ).fit_predict(features)


def length_constrained_shape_clustering(length_block, shape_block, stat_df, seed=cfg.RANDOM_STATE):
    """
    Final clustering algorithm.

    Stage 1: group motifs into length bands.
    Stage 2: for medium/long bands, split motifs into early-valley vs late-valley
    shape families using morphology + PCA shape features.
    """
    length_labels = KMeans(#reqemleri ver men oxsarlari bi yere
        n_clusters=cfg.N_LENGTH_BANDS,
        random_state=seed,
        n_init=30,
    ).fit_predict(length_block)

    # Rename bands so that band 0 is shortest and the last band is longest.
    band_order = sorted(np.unique(length_labels), key=lambda c: float(length_block[length_labels == c].mean()))
    band_map = {old: new for new, old in enumerate(band_order)}
    bands = np.array([band_map[x] for x in length_labels])

    labels = np.full(len(bands), -1, dtype=int)
    metadata = []
    next_cluster = 0

    for band in sorted(np.unique(bands)):
        idx = np.where(bands == band)[0]
        if band not in cfg.SPLIT_SHAPE_BANDS:
            labels[idx] = next_cluster
            metadata.append({
                "cluster": next_cluster,
                "length_band": int(band),
                "shape_family": "single",
            })
            next_cluster += 1
            continue

        # Use both interpretable morphology and PCA shape, but order the final split by valley position.
        local_stats = StandardScaler().fit_transform(stat_df.iloc[idx][SHAPE_SPLIT_COLS].to_numpy())
        local_shape = shape_block[idx, :5]
        local_features = np.hstack([local_stats * 1.2, local_shape * 0.6])
        sub_labels = KMeans(
            n_clusters=cfg.SHAPE_SPLIT_K,
            random_state=seed,
            n_init=30,
        ).fit_predict(local_features)

        # Stable semantic IDs: early-valley subgroup first, late-valley subgroup second.
        sub_order = sorted(
            np.unique(sub_labels),
            key=lambda s: float(stat_df.iloc[idx]["min_position"].to_numpy()[sub_labels == s].mean()),
        )
        for sub in sub_order:
            chosen = idx[sub_labels == sub]
            labels[chosen] = next_cluster
            avg_pos = float(stat_df.iloc[chosen]["min_position"].mean())
            metadata.append({
                "cluster": next_cluster,
                "length_band": int(band),
                "shape_family": "early-valley" if avg_pos < 0.5 else "late-valley",
            })
            next_cluster += 1

    if (labels < 0).any():
        raise RuntimeError("Some motifs were not assigned to a cluster.")
    return labels, pd.DataFrame(metadata)



def compute_lcsc_stability(length_block, shape_block, stat_df, reference_labels):
    """Estimate stability of the final LCSC method across random seeds."""
    scores = []
    for seed in range(cfg.RANDOM_STATE + 1, cfg.RANDOM_STATE + 6):
        labels_seed, _ = length_constrained_shape_clustering(length_block, shape_block, stat_df, seed=seed)
        scores.append(adjusted_rand_score(reference_labels, labels_seed))
    return float(np.mean(scores)), float(np.std(scores))

def evaluate_labels(features, labels, lengths, stat_df=None):
    """Compute clustering metrics. Silhouette is sampled for speed.

    full_feature_silhouette is strict and uses all engineered features.
    task_axis_silhouette uses the two explicit task axes: log-length and valley timing.

    
    Silhouette Score
    a = bu motif öz cluster-indəki digərlərinə nə qədər yaxındır?
    b = bu motif ən yaxın başqa cluster-ə nə qədər uzaqdır?
    score = (b - a) / max(a, b)
    """
    sample_size = min(cfg.SAMPLE_SIZE, len(labels))
    full_sil = float(silhouette_score(features, labels, sample_size=sample_size, random_state=cfg.RANDOM_STATE))
    task_sil = np.nan
    if stat_df is not None:
        task_axes = StandardScaler().fit_transform(
            np.column_stack([np.log1p(lengths), stat_df["min_position"].to_numpy()])
        )

        ''' Task Silhouette
        Eyni silhouette, amma yalnız 2 şeyə baxır:
        
        log_length — uzunluq
        min_position — dibin mövqeyi
        Bizim nəticə: 0.23 — çox daha yaxşı.
        Niyə bu daha vacibdir? 
        Çünki task məhz bu iki şeyi istəyir — uzunluq ayrımı və forma ayrımı. 
        Digər 20 feature-un overlap etməsi problem deyil'''
        task_sil = float(silhouette_score(task_axes, labels, sample_size=sample_size, random_state=cfg.RANDOM_STATE))
    return {
        "n_clusters": int(len(np.unique(labels))),
        "full_feature_silhouette": full_sil,
        "task_axis_silhouette": task_sil,
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "length_separation": float(length_separation_score(lengths, labels)),
        "min_cluster_share": float(min(np.bincount(labels)) / len(labels)),
    }


def compare_methods(features, length_block, shape_block, stat_df, lengths):
    """Evaluate baselines and final method side by side."""
    rows = []

    length_labels = length_only_baseline(length_block)
    rows.append({"method": "Length-only baseline", **evaluate_labels(features, length_labels, lengths, stat_df)})

    flat_labels = flat_kmeans_baseline(features)
    rows.append({"method": "Flat k-means baseline", **evaluate_labels(features, flat_labels, lengths, stat_df)})

    final_labels, metadata = length_constrained_shape_clustering(length_block, shape_block, stat_df)
    rows.append({"method": "Final LCSC method", **evaluate_labels(features, final_labels, lengths, stat_df)})

    return final_labels, metadata, pd.DataFrame(rows), length_labels, flat_labels


def main():
    start_time = time.time()
    print("=" * 76)
    print("MOTIF CLUSTERING PIPELINE: LENGTH-CONSTRAINED SHAPE CLUSTERING")
    print("=" * 76)

    print("\nSTEP 1: Loading data")
    df = pd.read_csv(cfg.DATA_PATH)
    print(f"  Data shape: {df.shape[0]} time steps x {df.shape[1]} signals")

    print("\nSTEP 2: Extracting motifs")
    motif_rows, raw_segments, resampled_shapes, stat_rows = extract_all_motifs(df, cfg)
    motif_df = pd.DataFrame(motif_rows)
    stat_df = pd.DataFrame(stat_rows)
    lengths = motif_df["length"].to_numpy()
    print(f"  Motifs extracted: {len(motif_df)}")
    print(f"  Length range: {lengths.min()} to {lengths.max()} samples")

    print("\nSTEP 3: Building feature representation")
    shape_block, stats_block, length_block, explained_var = build_feature_blocks(resampled_shapes, stat_df)
    features = task_feature_space(shape_block, stats_block, length_block)
    print(f"  PCA shape variance retained: {explained_var * 100:.2f}%")
    print("  Feature groups: shape PCA + morphology statistics + log-length")

    print("\nSTEP 4: Comparing baselines with final method")
    labels, cluster_meta, comparison_df, length_labels, flat_labels = compare_methods(
        features, length_block, shape_block, stat_df, lengths
    )
    final_metrics = comparison_df[comparison_df["method"] == "Final LCSC method"].iloc[0].to_dict()
    stability = compute_lcsc_stability(length_block, shape_block, stat_df, labels)
    comparison_df.loc[comparison_df["method"] == "Final LCSC method", "stability_ari_mean"] = stability[0]
    comparison_df.loc[comparison_df["method"] == "Final LCSC method", "stability_ari_std"] = stability[1]
    print(comparison_df.to_string(index=False))
    print(f"  Stability ARI={stability[0]:.4f} +/- {stability[1]:.4f}")

    print("\nSTEP 5: Summaries and exports")
    motif_df["cluster"] = labels.astype(int)
    results_df = pd.concat([motif_df, stat_df], axis=1)
    summary_df = build_cluster_summary(motif_df.drop(columns=["cluster"]), stat_df, labels, cluster_meta)
    cluster_names = assign_cluster_names(summary_df)
    summary_df["cluster_name"] = summary_df["cluster"].map(cluster_names)

    results_df.to_csv(cfg.BASE_DIR / "motif_results.csv", index=False)
    summary_df.to_csv(cfg.BASE_DIR / "cluster_summary.csv", index=False)
    comparison_df.to_csv(cfg.BASE_DIR / "model_comparison_scores.csv", index=False)
    comparison_df.to_csv(cfg.BASE_DIR / "evaluation_metrics.csv", index=False)

    write_methodology_report(
        cfg.BASE_DIR / "methodology_report.md",
        cfg,
        final_metrics,
        summary_df,
        comparison_df,
        cluster_names,
        stability,
        explained_var,
    )

    for _, row in summary_df.iterrows():
        print(
            f"    Cluster {int(row['cluster'])}: {row['cluster_name']} | "
            f"n={int(row['count'])}, len=[{int(row['min_length'])},{int(row['max_length'])}], "
            f"avg_len={row['avg_length']}, valley={row['avg_min_position']}"
        )

    print("\nSTEP 6: Figures")
    save_method_comparison(comparison_df, cfg.BASE_DIR / "fig1_method_comparison.png")
    save_cluster_prototypes(
        resampled_shapes, labels, lengths, summary_df, cluster_names,
        cfg.BASE_DIR / "fig2_cluster_prototypes.png", cfg.RANDOM_STATE
    )
    save_feature_space(features, labels, cfg.BASE_DIR / "fig3_feature_space.png", cfg.RANDOM_STATE)
    save_length_distribution(lengths, labels, summary_df, cfg.BASE_DIR / "fig4_length_distribution.png")
    save_cluster_profile(summary_df, cfg.BASE_DIR / "fig5_cluster_profile.png")
    save_sample_signals(df, motif_df.drop(columns=["cluster"]), labels, cfg, cfg.BASE_DIR / "fig6_signal_examples.png")
    print("  Saved 6 figures")

    runtime = time.time() - start_time
    print(f"\nCompleted in {runtime:.2f} seconds")


if __name__ == "__main__":
    main()
