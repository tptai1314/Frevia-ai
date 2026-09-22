"""Ranking metrics & statistical tests (paper 2026 protocol)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-query precision@k: top-k retrieved items belonging to target class."""
    if k <= 0:
        raise ValueError('k must be positive')
    out = np.zeros(scores.shape[0], dtype=float)
    for i in range(scores.shape[0]):
        top_k = labels[i][np.argsort(-scores[i])[:k]]
        out[i] = top_k.sum() / k
    return out


def recall_at_k(
    scores: np.ndarray, labels: np.ndarray, k: int = 10
) -> np.ndarray:
    """Per-query recall@k: |top-k ∩ relevant| / total relevant."""
    out = np.zeros(scores.shape[0], dtype=float)
    total = labels.sum(axis=1)
    for i in range(scores.shape[0]):
        if total[i] == 0:
            continue
        top_k = labels[i][np.argsort(-scores[i])[:k]]
        out[i] = top_k.sum() / total[i]
    return out


def summarize(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test (one-sided: a > b)."""
    stat, p = sp_stats.wilcoxon(a, b, alternative='greater')
    return float(stat), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    diff = a - b
    std = float(diff.std(ddof=1))
    if std == 0:
        return float('inf') if diff.mean() > 0 else 0.0
    return float(diff.mean() / std)