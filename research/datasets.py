"""Dataset loading for FREVIA research (paper 2026 reproduction)."""

import pandas as pd

from . import config


def load_cleaned_jobs() -> pd.DataFrame:
    """Glassdoor Data Science job postings (660 in paper 2026)."""
    df = pd.read_csv(config.CLEANED_JOBS_CSV, encoding='utf-8')
    df = df.dropna(subset=['Job Description']).copy()
    df['job_id'] = df.index
    df['text'] = df['Job Description'].astype(str)
    return df


def load_resumes() -> pd.DataFrame:
    """Kaggle Resume Dataset (962 profiles, 25 domains in paper 2026)."""
    df = pd.read_csv(config.RESUMES_CSV, encoding='utf-8')
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Resume', 'Category']).copy()
    df['resume_id'] = df.index
    df['text'] = df['Resume'].astype(str)
    return df


def build_eval_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluation pool per paper 2026: Data Science (40) + Hadoop (42)."""
    return df[df['Category'].isin(config.TARGET_DOMAINS)].reset_index(drop=True)