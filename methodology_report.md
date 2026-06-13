# Motif Clustering Methodology Report

## Executive Summary
This version uses a length-constrained shape clustering strategy. The earlier flat k-means approach was cleaner mathematically, but it mixed very different motif lengths inside the same cluster. The task explicitly states that strongly different lengths should be separated, so the final method first creates length bands and then performs shape clustering inside the medium and long bands.

## 1. Motif Extraction
- Adaptive thresholding: `True`
- Global threshold suggested by task: `110.0` used as reference baseline.
- Final threshold per signal = percentile(80) - 10.0
- Minimum motif length = 20
- Internal gaps up to 5 samples are merged.

Reasoning: values around 120 are steady background. A fixed threshold of 110 is a useful first estimate, but a per-signal threshold is safer because not all columns have exactly the same baseline level.

## 2. Feature Design
- Normalized motif curves are resampled to 120 points.
- PCA keeps 10 components and explains 98.13% of normalized shape variance.
- Morphological features include depth, area, valley position, half-width, slopes, skewness, and kurtosis.
- Length is represented as log-length to reduce the effect of extreme long motifs.

## 3. Final Clustering Method: LCSC
LCSC = Length-Constrained Shape Clustering.

1. Cluster motif log-length into 4 ordered length bands.
2. Keep very short and short bands as compact length groups.
3. Split medium and long bands into early-valley and late-valley shape families.
4. Assign stable semantic cluster IDs ordered by length first, then valley position.

This directly addresses the main review issue: motifs with very different temporal lengths are no longer mixed in the same final cluster.

## 4. Baseline Comparison
| method                |   n_clusters |   full_feature_silhouette |   task_axis_silhouette |   davies_bouldin |   calinski_harabasz |   length_separation |   min_cluster_share |   stability_ari_mean |   stability_ari_std |
|:----------------------|-------------:|--------------------------:|-----------------------:|-----------------:|--------------------:|--------------------:|--------------------:|---------------------:|--------------------:|
| Length-only baseline  |            4 |                    0.042  |                 0.1884 |           3.8852 |             744.198 |              0.8504 |              0.1481 |             nan      |            nan      |
| Flat k-means baseline |            6 |                    0.0933 |                 0.0553 |           2.7124 |             894.485 |              0.3485 |              0.0714 |             nan      |            nan      |
| Final LCSC method     |            6 |                    0.0304 |                 0.2316 |           3.7591 |             576.723 |              0.8505 |              0.0694 |               0.9893 |              0.0101 |

Important interpretation: the final method is selected for task fit, not for maximum full-feature silhouette. The data behaves more like a continuum than cleanly separated islands, so the strict full-feature silhouette remains modest. However, the task-axis silhouette, length separation, and seed stability are much stronger.

## 5. Final Evaluation
- Final number of clusters: 6
- Full-feature silhouette score: 0.0304
- Task-axis silhouette score (length + valley timing): 0.2316
- Davies-Bouldin score: 3.7591
- Calinski-Harabasz score: 576.72
- Length separation score: 0.8505
- Stability ARI across seeds: 0.9893 ± 0.0101

## 6. Cluster Interpretation
- Cluster 0 — Very short compact motif: n=1100, length=[20, 36], avg_length=25.89, avg_depth=23.66, valley_position=0.488
- Cluster 1 — Short compact motif: n=1598, length=[37, 68], avg_length=52.31, avg_depth=70.15, valley_position=0.463
- Cluster 2 — Medium early valley motif: n=1808, length=[69, 117], avg_length=92.1, avg_depth=84.5, valley_position=0.282
- Cluster 3 — Medium late valley motif: n=1470, length=[69, 117], avg_length=91.88, avg_depth=85.52, valley_position=0.638
- Cluster 4 — Long early valley motif: n=552, length=[118, 292], avg_length=157.0, avg_depth=94.17, valley_position=0.276
- Cluster 5 — Long late valley motif: n=487, length=[118, 302], avg_length=154.65, avg_depth=93.36, valley_position=0.643

## 7. Honest Limitations
- Full-feature silhouette is still low, so the motifs should be understood as overlapping families rather than perfectly separated natural classes.
- The shape split mainly separates early-valley versus late-valley patterns within length bands. More complex shape types may require DTW or k-shape.
- Extraction quality should ideally be validated by manually reviewing 30-50 motifs.
- A production version should include a noise/outlier cluster for borderline short segments.
