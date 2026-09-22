"""Run FREVIA varians vs baselines on the evaluation protocol.

FREVIA-min:   multi-view (4 same-view signals, uniform).
FREVIA-cross: + cross-view signal (uniform over 5 signals).
FREVIA-full:  adaptive fusion (per-job learned gating).

Skills like keyword/TF-IDF/SBERT are recomputed on the same test jobs.

Usage:
    python -m research.run_frevia --jobs-sample 200 --epochs 20 [--no-adaptive]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config, datasets, evaluate, frevia, matching

SBERT_MODEL = 'all-MiniLM-L6-v2'


def _labels(pool: pd.DataFrame, domain: str) -> np.ndarray:
    return (pool['Category'] == domain).to_numpy().astype(int)


def run(
    jobs_sample: int | None = None,
    epochs: int = 20,
    seed: int = 42,
    do_adaptive: bool = True,
) -> pd.DataFrame:
    np.random.seed(seed)

    jobs_df = datasets.load_cleaned_jobs()
    resumes_all = datasets.load_resumes()
    pool = datasets.build_eval_pool(resumes_all)

    job_views = frevia.load_job_views()
    resume_views = frevia.load_resume_views()

    jobs_df = jobs_df[jobs_df.index.isin(job_views)].copy()
    pool = pool[pool.index.isin(resume_views)].copy()

    if jobs_sample is not None and jobs_sample < len(jobs_df):
        jobs_df = jobs_df.sample(jobs_sample, random_state=seed)

    n_jobs = len(jobs_df)
    n_res = len(pool)
    print(f'jobs={n_jobs} resumes={n_res}')

    job_views = {i: job_views[i] for i in jobs_df.index}

    # train/test split over jobs (adaptive weights generalize to unseen jobs)
    job_ids = list(jobs_df.index)
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(n_jobs, dtype=bool)
    test_idx = rng.choice(n_jobs, size=max(1, n_jobs // 2), replace=False)
    test_mask[test_idx] = True
    train_idx = ~test_mask

    model = SentenceTransformer(SBERT_MODEL)
    job_ids_order, job_emb = frevia.embed_views(job_views, model)
    res_ids_order, res_emb = frevia.embed_views(resume_views, model)

    align = {rid: i for i, rid in enumerate(res_ids_order)}
    res_order = [align[i] for i in pool.index]
    res_emb = res_emb[res_order]
    label_ds = _labels(pool, 'Data Science')
    label_hadoop = _labels(pool, 'Hadoop')

    signals = frevia.score_signals(job_emb, res_emb)
    print('signals computed:', signals.shape)

    # --- adaptive fusion training (on train jobs) ---
    adaptive_model = None
    if do_adaptive:
        print('Training adaptive fusion...')
        job_emb_train = job_emb[train_idx]
        signals_train = signals[train_idx]
        labels_train = np.repeat(label_ds[None, :], train_idx.sum(), axis=0)
        adaptive_model = frevia.train_adaptive(
            job_emb_train, signals_train, labels_train, epochs=epochs, seed=seed
        )

    test_jobs = jobs_df.iloc[test_idx]
    test_pos = np.where(test_mask)[0]

    method_scores: dict[str, np.ndarray] = {}

    # FREVIA static variants (test jobs)
    method_scores['frevia-min'] = frevia.static_score(
        signals[test_pos], [0.25, 0.25, 0.25, 0.25, 0.0]
    )
    method_scores['frevia-cross'] = frevia.static_score(
        signals[test_pos], [0.2, 0.2, 0.2, 0.2, 0.2]
    )
    if adaptive_model is not None:
        method_scores['frevia-full'] = frevia.predict_adaptive(
            adaptive_model, job_emb[test_pos], signals[test_pos]
        )

    # baselines on the same test jobs (whole-document representations)
    job_test_texts = test_jobs['text'].tolist()
    res_texts = pool['text'].tolist()
    method_scores['keyword'] = matching.keyword_similarity(
        job_test_texts, res_texts
    )
    method_scores['tfidf'] = matching.tfidf_cosine_similarity(
        job_test_texts, res_texts
    )
    res_emb_full = model.encode(res_texts, batch_size=64, show_progress_bar=False)
    job_emb_full = model.encode(
        job_test_texts, batch_size=64, show_progress_bar=False
    )
    from sklearn.metrics.pairwise import cosine_similarity

    method_scores['sbert'] = cosine_similarity(job_emb_full, res_emb_full)

    # evaluation
    rows: list[dict[str, float]] = []
    for name, sc in method_scores.items():
        for domain, label in (('Data Science', label_ds), ('Hadoop', label_hadoop)):
            labels = np.repeat(label[None, :], sc.shape[0], axis=0)
            p = evaluate.precision_at_k(sc, labels, config.TOP_K)
            r = evaluate.recall_at_k(sc, labels, config.TOP_K)
            rows.append(
                {
                    'method': name,
                    'domain': domain,
                    'P@10_mean': p.mean(),
                    'P@10_std': p.std(ddof=1),
                    'R@10_mean': r.mean(),
                    'R@10_std': r.std(ddof=1),
                    'n_jobs': sc.shape[0],
                }
            )
            print(
                f'[{name:>11s}] {domain:>14s}: '
                f'P@10={p.mean():.4f}+-{p.std(ddof=1):.4f}  '
                f'R@10={r.mean():.4f}+-{r.std(ddof=1):.4f}'
            )

    out = pd.DataFrame(rows)
    out.to_csv(config.PROCESSED_DIR / 'frevia_results.csv', index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Run FREVIA vs baselines.')
    parser.add_argument('--jobs-sample', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-adaptive', action='store_true')
    args = parser.parse_args()
    run(
        jobs_sample=args.jobs_sample,
        epochs=args.epochs,
        seed=args.seed,
        do_adaptive=not args.no_adaptive,
    )


if __name__ == '__main__':
    main()