"""Reproduce paper 2026 baselines on the evaluation protocol.

Usage:
    python -m research.run_baseline_2026 --methods keyword tfidf [--jobs-sample 660]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config, datasets, evaluate, matching


def _one_hot_labels(pool: pd.DataFrame, domain: str) -> np.ndarray:
    return (pool['Category'] == domain).to_numpy().astype(int).reshape(1, -1)


def _score_series(
    scores: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    p = evaluate.precision_at_k(scores, np.repeat(labels, scores.shape[0], axis=0), k)
    r = evaluate.recall_at_k(scores, np.repeat(labels, scores.shape[0], axis=0), k)
    return p, r


def run(
    methods: list[str],
    jobs_sample: int | None = None,
    k: int = config.TOP_K,
    pool: str = 'full',
) -> pd.DataFrame:
    jobs = datasets.load_cleaned_jobs()
    resumes = datasets.load_resumes()
    pool_df = (
        datasets.build_eval_pool(resumes)
        if pool == 'paper'
        else resumes.reset_index(drop=True)
    )

    if jobs_sample is not None and jobs_sample < len(jobs):
        jobs = jobs.sample(jobs_sample, random_state=42).reset_index(drop=True)

    job_texts = jobs['text'].tolist()
    resume_texts = pool_df['text'].tolist()

    scores: dict[str, np.ndarray] = {}
    if 'keyword' in methods:
        scores['keyword'] = matching.keyword_similarity(job_texts, resume_texts)
    if 'tfidf' in methods:
        scores['tfidf'] = matching.tfidf_cosine_similarity(job_texts, resume_texts)
    if 'sbert' in methods:
        scores['sbert'] = matching.sbert_cosine_similarity(job_texts, resume_texts)

    rows: list[dict[str, float]] = []
    domains = {
        d: _one_hot_labels(pool_df, d) for d in config.TARGET_DOMAINS
    }

    for name, mat in scores.items():
        for domain, labels in domains.items():
            p_series, r_series = _score_series(mat, labels, k)
            p_mean, p_std = evaluate.summarize(p_series)
            r_mean, r_std = evaluate.summarize(r_series)
            rows.append(
                {
                    'method': name,
                    'domain': domain,
                    'P@10_mean': p_mean,
                    'P@10_std': p_std,
                    'R@10_mean': r_mean,
                    'R@10_std': r_std,
                    'n_jobs': mat.shape[0],
                }
            )
            print(
                f'[{name:>8s}] {domain:>14s}: '
                f'P@10={p_mean:.4f}+-{p_std:.4f}  R@10={r_mean:.4f}+-{r_std:.4f} '
                f'({mat.shape[0]} jobs)'
            )

    if {'keyword', 'tfidf'} <= set(scores):
        diff = _paired_p10(scores['tfidf'], scores['keyword'], domains)
        zeros = np.zeros_like(diff)
        _stat, p_wilcoxon = evaluate.wilcoxon_signed_rank(diff, zeros)
        d_cohen = evaluate.cohens_d(diff, zeros)
        print(
            '\nWilcoxon (TF-IDF P@10 > Keyword P@10, pooled domains): '
            f'p={p_wilcoxon:.3e} | Cohen\'s d={d_cohen:.3f}'
        )

    out = pd.DataFrame(rows)
    out.to_csv(
        config.PROCESSED_DIR / f'baseline_2026_results_{pool}.csv', index=False
    )
    return out


def _paired_p10(a: np.ndarray, b: np.ndarray, domains: dict) -> np.ndarray:
    """Paired per-query P@10 series pooled across target domains (a vs b)."""
    series = []
    for labels in domains.values():
        pa = evaluate.precision_at_k(a, np.repeat(labels, a.shape[0], axis=0))
        pb = evaluate.precision_at_k(b, np.repeat(labels, b.shape[0], axis=0))
        series.append(pa - pb)
    return np.concatenate(series)


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper 2026 baselines.')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['keyword', 'tfidf'],
        choices=['keyword', 'tfidf', 'sbert'],
    )
    parser.add_argument(
        '--jobs-sample',
        type=int,
        default=None,
        help='Only evaluate on a random sample of N job postings (None = all).',
    )
    parser.add_argument(
        '--pool',
        choices=['paper', 'full'],
        default='full',
        help='paper = 82 CV (DS+Hadoop) per paper; full = 962 CV / 25 domains.',
    )
    args = parser.parse_args()
    run(methods=args.methods, jobs_sample=args.jobs_sample, pool=args.pool)


if __name__ == '__main__':
    main()