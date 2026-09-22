"""FREVIA model: multi-view representation, cross-view interaction, adaptive fusion.

Views: title, skills, experience, projects.

Signals per (job, resume) pair:
- 4 same-view signals: cos(view_job_v, view_resume_v)
- 1 cross-view signal: mean cosine over predefined cross-view pairs.

Fusion:
- static: fixed/uniform weights.
- adaptive: per-job weights alpha(j) = softmax(MLP(job_view_embeddings)),
  trained with pairwise (BPR) ranking loss on domain labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import config

VIEWS = ('title', 'skills', 'experience', 'projects')

CROSS_PAIRS = (
    ('skills', 'experience'),
    ('skills', 'projects'),
    ('experience', 'projects'),
    ('projects', 'experience'),
    ('title', 'skills'),
    ('experience', 'skills'),
)

N_SIGNALS = len(VIEWS) + 1  # 4 same-view + 1 cross-view

CROSS_IDX = len(VIEWS)


def _view_index(name: str) -> int:
    return VIEWS.index(name)


def load_views(path: Path) -> dict[int, dict[str, Any]]:
    """Load {doc_id: views} from a checkpointed JSONL file."""
    out: dict[int, dict[str, Any]] = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            row = json.loads(line)
            if row.get('ok'):
                out[int(row['id'])] = row['views']
    return out


def load_resume_views() -> dict[int, dict[str, Any]]:
    return load_views(config.PROCESSED_DIR / 'views_resumes.jsonl')


def load_job_views() -> dict[int, dict[str, Any]]:
    return load_views(config.PROCESSED_DIR / 'views_jobs.jsonl')


def view_to_text(view: dict[str, Any], name: str) -> str:
    if name == 'title':
        return str(view.get('title') or '')
    if name == 'skills':
        return ' '.join(s for s in view.get('skills', []) if isinstance(s, str))
    parts: list[str] = []
    for item in view.get(name, []) or []:
        if not isinstance(item, dict):
            continue
        seg = []
        for key in ('position', 'name', 'description'):
            if item.get(key):
                seg.append(str(item[key]))
        techs = [t for t in item.get('technologies', []) if isinstance(t, str)]
        if techs:
            seg.append(' '.join(techs))
        years = item.get('years')
        if years:
            seg.append(f'{years} years')
        parts.append(' '.join(seg))
    return ' '.join(parts)


def embed_views(
    docs: dict[int, dict[str, Any]], model: Any
) -> tuple[list[int], np.ndarray]:
    """Return (ids, embeddings) where embeddings[i] shape is (len(VIEWS), dim)."""
    ids = list(docs.keys())
    texts_by_view: list[list[str]] = []
    for name in VIEWS:
        texts_by_view.append([view_to_text(docs[i], name) for i in ids])
    # encode: each view separately, then stack
    encoded = []
    for texts in texts_by_view:
        encoded.append(model.encode(texts, batch_size=64, show_progress_bar=False))
    emb = np.stack(encoded, axis=1)  # (n_docs, 4, d)
    return ids, emb


def score_signals(
    job_emb: np.ndarray, resume_emb: np.ndarray
) -> np.ndarray:
    """n_jobs x n_resumes x n_signals cosine-similarity signals."""
    n_jobs, n_views, _d = job_emb.shape
    n_res = resume_emb.shape[0]
    job_norm = job_emb / np.linalg.norm(job_emb, axis=-1, keepdims=True)
    res_norm = resume_emb / np.linalg.norm(resume_emb, axis=-1, keepdims=True)

    # sim[i, a, b, r] = cosine(job view a of job i, resume view b of resume r)
    sim = np.zeros((n_jobs, n_views, n_views, n_res), dtype=float)
    for a in range(n_views):
        for b in range(n_views):
            sim[:, a, b, :] = job_norm[:, a, :] @ res_norm[:, b, :].T

    signals = np.zeros((n_jobs, n_res, N_SIGNALS), dtype=float)
    for v in range(n_views):
        signals[:, :, v] = sim[:, v, v, :]
    cross = np.zeros((n_jobs, n_res), dtype=float)
    for i_name, j_name in CROSS_PAIRS:
        cross += sim[:, _view_index(i_name), _view_index(j_name), :]
    signals[:, :, CROSS_IDX] = cross / len(CROSS_PAIRS)
    return signals


def static_score(signals: np.ndarray, weights: list[float]) -> np.ndarray:
    """Linear fusion with fixed weights -> (n_jobs, n_res)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return signals @ w


class AdaptiveFusion(nn.Module):
    """Per-job weight gating: alpha(job) = softmax(MLP(job view embeddings))."""

    def __init__(self, embed_dim: int, n_signals: int = N_SIGNALS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * len(VIEWS), 64),
            nn.ReLU(),
            nn.Linear(64, n_signals),
        )

    def forward(self, job_emb: torch.Tensor) -> torch.Tensor:
        x = job_emb.reshape(job_emb.shape[0], -1)
        return torch.softmax(self.net(x), dim=-1)


def train_adaptive(
    job_emb: np.ndarray,
    signals: np.ndarray,
    labels: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 0,
) -> AdaptiveFusion:
    """Train per-job gating with BPR. labels: (n_jobs, n_resumes) binary."""
    torch.manual_seed(seed)
    n_jobs, n_res, n_sig = signals.shape
    model = AdaptiveFusion(job_emb.shape[-1], n_signals=N_SIGNALS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    job_t = torch.tensor(job_emb, dtype=torch.float32)
    sig_t = torch.tensor(signals, dtype=torch.float32)

    for epoch in range(epochs):
        total_loss = 0.0
        perm = np.random.permutation(n_jobs)
        for j in perm:
            pos_mask = labels[j] == 1
            if not pos_mask.any():
                continue
            neg_mask = labels[j] == 0
            r_pos = int(np.random.choice(np.where(pos_mask)[0]))
            if neg_mask.any():
                r_neg = int(np.random.choice(np.where(neg_mask)[0]))
            else:
                r_neg = r_pos
            alpha = model(job_t[j:j + 1])  # (1, n_signals)
            score_pos = (alpha * sig_t[j, r_pos]).sum()
            score_neg = (alpha * sig_t[j, r_neg]).sum()
            loss = -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch in (0, epochs // 2, epochs - 1):
            print(f'  [adaptive] epoch {epoch + 1:>2d}/{epochs}  loss={total_loss:.4f}')
    return model


def predict_adaptive(
    model: AdaptiveFusion, job_emb: np.ndarray, signals: np.ndarray
) -> np.ndarray:
    """Score with trained gating: sum over all weighted signals."""
    with torch.no_grad():
        w = model(torch.tensor(job_emb, dtype=torch.float32))  # (n_jobs, n_signals)
    w = w.numpy()
    scores = np.zeros((signals.shape[0], signals.shape[1]), dtype=float)
    for v in range(N_SIGNALS):
        scores += w[:, v:v + 1] * signals[:, :, v]
    return scores