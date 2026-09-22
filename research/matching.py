"""Baseline matching methods (paper 2026).

- Keyword-based: raw token overlap ratio on preprocessed text.
- TF-IDF: whole-document TF-IDF (unigram, L2-normalized) + cosine similarity.
- SBERT: sentence-transformer embeddings + cosine similarity (optional).
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import preprocess


def _overlap_ratio(jd_tokens: set[str], resume_tokens: set[str]) -> float:
    if not jd_tokens or not resume_tokens:
        return 0.0
    return len(jd_tokens & resume_tokens) / len(jd_tokens | resume_tokens)


def keyword_similarity(job_texts: list[str], resume_texts: list[str]) -> np.ndarray:
    """Raw keyword overlap ratio between each job and each resume."""
    job_sets = [set(preprocess.preprocess_text(t).split()) for t in job_texts]
    resume_sets = [set(preprocess.preprocess_text(t).split()) for t in resume_texts]
    matrix = np.zeros((len(job_sets), len(resume_sets)), dtype=float)
    for i, js in enumerate(job_sets):
        for j, rs in enumerate(resume_sets):
            matrix[i, j] = _overlap_ratio(js, rs)
    return matrix


def tfidf_cosine_similarity(job_texts: list[str], resume_texts: list[str]) -> np.ndarray:
    """Whole-document TF-IDF (unigram, L2) + cosine similarity."""
    corpus = list(job_texts) + list(resume_texts)
    corpus = [preprocess.preprocess_text(t) for t in corpus]
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), norm='l2')
    tfidf = vectorizer.fit_transform(corpus)
    jobs = tfidf[: len(job_texts)]
    resumes = tfidf[len(job_texts):]
    return cosine_similarity(jobs, resumes)


def sbert_cosine_similarity(
    job_texts: list[str], resume_texts: list[str], model_name: str = 'all-MiniLM-L6-v2'
) -> np.ndarray:
    """SBERT sentence embeddings + cosine similarity (optional, heavy)."""
    from sentence_transformers import SentenceTransformer  # lazy import

    model = SentenceTransformer(model_name)
    job_emb = model.encode(job_texts, batch_size=64, show_progress_bar=True)
    resume_emb = model.encode(resume_texts, batch_size=64, show_progress_bar=True)
    return cosine_similarity(job_emb, resume_emb)


def greedy_one_to_one(scores: np.ndarray) -> np.ndarray:
    """Greedy one-to-one matching: pick highest remaining (job, resume) pair."""
    pairs: list[tuple[int, int]] = []
    used_jobs: set[int] = set()
    used_resumes: set[int] = set()
    order = np.dstack(
        np.unravel_index(
            np.argsort(-scores, axis=None), scores.shape
        )
    )[0].tolist()
    for i, j in order:
        if i in used_jobs or j in used_resumes:
            continue
        used_jobs.add(i)
        used_resumes.add(j)
        pairs.append((int(i), int(j)))
    assignment = np.zeros(scores.shape, dtype=bool)
    for i, j in pairs:
        assignment[i, j] = True
    return assignment