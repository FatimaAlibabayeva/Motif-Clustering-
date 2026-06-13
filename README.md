# Motifs Clustering in Time Series Data

Author: **Fatima Alibabayeva**  
Task: **Algorithm Developer Task 2026**

## Solution overview

This project extracts candidate motifs from 500 time-series signals and groups them using **Length-Constrained Shape Clustering (LCSC)**.

The main design idea is simple: the task defines motif similarity using both **duration** and **shape**. Therefore, the pipeline first separates motifs into major length ranges, then uses interpretable shape characteristics to split medium and long motifs into meaningful families.

## Final cluster families

| Cluster | Interpretation |
|---|---|
| C0 | Very short compact motif |
| C1 | Short compact motif |
| C2 | Medium early-valley motif |
| C3 | Medium late-valley motif |
| C4 | Long early-valley motif |
| C5 | Long late-valley motif |

## How to run

```bash
pip install -r requirements.txt
python motif_clustering.py
```

Expected runtime on a typical laptop: about **15-20 seconds**.

## Main outputs

| File | Purpose |
|---|---|
| `motif_results.csv` | Every extracted motif with signal name, start/end indices, cluster ID, and features |
| `cluster_summary.csv` | Interpretable cluster-level statistics |
| `evaluation_metrics.csv` | Baseline comparison and final evaluation metrics |
| `model_comparison_scores.csv` | Metric table saved for reporting |
| `methodology_report.md` | Detailed explanation and possible next improvements |
| `fig1_method_comparison.png` | Baseline vs final method comparison |
| `fig2_cluster_prototypes.png` | Generalized motif prototypes per cluster |
| `fig3_feature_space.png` | 2D projection of feature space |
| `fig4_length_distribution.png` | Length distribution by cluster |
| `fig5_cluster_profile.png` | Cluster profile across interpretable features |
| `fig6_signal_examples_clean.png` | Readable signal-level validation examples |
| `motif_clustering_INTERVIEW_CONFIDENT_presentation.pptx` | Interview presentation deck |

## Evaluation note

The final method is selected for **task alignment**, **length consistency**, **interpretability**, and **reproducibility**. Since the motifs behave like overlapping families rather than perfectly separated natural classes, the evaluation uses both standard clustering metrics and task-specific validation checks.
