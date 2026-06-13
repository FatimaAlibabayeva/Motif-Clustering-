"""Reusable processing utilities for motif extraction and feature engineering."""

import numpy as np
from scipy.stats import skew, kurtosis


def estimate_signal_threshold(signal, steady_percentile, threshold_offset):
    """Estimate a per-signal threshold from the steady region near 120."""
    steady_level = np.percentile(signal, steady_percentile)
    return steady_level - threshold_offset, steady_level


def fill_short_gaps(mask, max_gap):
    """Merge tiny non-motif holes inside below-threshold motif regions."""
    if max_gap <= 0:
        return mask.copy()

    filled = mask.copy()
    n = len(filled)
    i = 0
    while i < n:
        if filled[i]:
            i += 1
            continue
        start = i
        while i < n and not filled[i]:
            i += 1
        end = i
        is_internal_gap = start > 0 and end < n and filled[start - 1] and filled[end]
        if is_internal_gap and (end - start) <= max_gap:
            filled[start:end] = True
    return filled


def extract_motifs(signal, threshold, min_len, max_gap):
    """Return contiguous below-threshold motif candidate regions."""
    below = fill_short_gaps(signal < threshold, max_gap)
    transitions = np.diff(np.r_[False, below, False].astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    return [(s, e, signal[s:e]) for s, e in zip(starts, ends) if (e - s) >= min_len]


def normalize_minmax(segment):
    """Normalize one motif to [0, 1] so shape is comparable across amplitudes."""
    mn, mx = np.min(segment), np.max(segment)
    return (segment - mn) / (mx - mn + 1e-8)


def resample_fixed(segment, fixed_len):
    """Resample a motif to a common length using linear interpolation."""
    x_old = np.linspace(0, 1, len(segment))
    x_new = np.linspace(0, 1, fixed_len)
    return np.interp(x_new, x_old, segment)


def motif_shape_statistics(segment, baseline):
    """Compute interpretable shape descriptors from the original motif."""
    segment = segment.astype(float)
    n = len(segment)
    mn, mx = float(np.min(segment)), float(np.max(segment))
    min_idx = int(np.argmin(segment))
    amplitude = mx - mn
    depth = float(baseline - mn)
    area_below_baseline = float(np.mean(np.maximum(baseline - segment, 0)))
    half_level = mn + 0.5 * amplitude
    half_width_fraction = float(np.mean(segment < half_level))
    min_position = float(min_idx / max(n - 1, 1))
    descent_slope = float((segment[min_idx] - segment[0]) / max(min_idx + 1, 1))
    recovery_slope = float((segment[-1] - segment[min_idx]) / max(n - min_idx, 1))
    start_end_delta = float(segment[0] - segment[-1])

    return {
        "log_length": float(np.log1p(n)),
        "depth": depth,
        "amplitude": float(amplitude),
        "area_below_baseline": area_below_baseline,
        "min_position": min_position,
        "half_width_fraction": half_width_fraction,
        "descent_slope": descent_slope,
        "recovery_slope": recovery_slope,
        "start_end_delta": start_end_delta,
        "std_value": float(np.std(segment)),
        "skewness": float(skew(segment)),#motifdeki dibin posititioni(evvel or axir)
        "kurtosis": float(kurtosis(segment)),#motif dibde ne qede qaldi,kicik kurtosos ymru dib daha cox time
    }


def extract_all_motifs(df, cfg):
    """Extract motifs from all 500 signals and return metadata, raw segments, resampled shapes, and stats."""
    motif_rows = []
    raw_segments = []
    resampled_shapes = []
    stat_rows = []

    for signal_name in df.columns:
        signal = df[signal_name].to_numpy(dtype=float)
        if cfg.USE_ADAPTIVE_THRESHOLD:
            threshold, baseline = estimate_signal_threshold(
                signal, cfg.STEADY_PERCENTILE, cfg.THRESHOLD_OFFSET
            )
        else:
            threshold = cfg.GLOBAL_THRESHOLD
            baseline = np.percentile(signal, cfg.STEADY_PERCENTILE)

        for start, end, segment in extract_motifs(signal, threshold, cfg.MIN_LEN, cfg.MAX_GAP):
            stats = motif_shape_statistics(segment, baseline)
            motif_rows.append({
                "signal": signal_name,
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "threshold": float(threshold),
                "baseline": float(baseline),
                "min_value": float(np.min(segment)),
                "mean_value": float(np.mean(segment)),
            })
            raw_segments.append(segment)
            resampled_shapes.append(resample_fixed(normalize_minmax(segment), cfg.FIXED_LEN))
            stat_rows.append(stats)

    if not motif_rows:
        raise RuntimeError("No motifs detected. Tune threshold/min_len parameters.")

    return motif_rows, raw_segments, np.asarray(resampled_shapes), stat_rows


def length_separation_score(length_values, labels):
    """Between-cluster length variance divided by total length variance."""
    length_values = np.asarray(length_values)
    total_var = np.var(length_values) + 1e-8
    global_mean = np.mean(length_values)
    between = 0.0
    for cluster_id in np.unique(labels):
        vals = length_values[labels == cluster_id]
        between += len(vals) * (np.mean(vals) - global_mean) ** 2
    return float(between / (len(length_values) * total_var))
