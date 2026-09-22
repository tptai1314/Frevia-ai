"""Extract FREVIA semantic views (title/skills/experience/projects) via Qwen.

Checkpointed JSONL output in data/processed/views_*.jsonl, so long runs can
resume after interruptions.

Usage:
    python -m research.run_extract_views --source resumes --limit 3
    python -m research.run_extract_views --source both --workers 3
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from . import config, datasets, extraction

_LOCK = threading.Lock()
_FIELDS = ('title', 'skills', 'experience', 'projects')


def _out_path(source: str) -> str:
    return str(config.PROCESSED_DIR / f'views_{source}.jsonl')


def _load_done(source: str) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    try:
        with open(_out_path(source), encoding='utf-8') as fh:
            for line in fh:
                row = json.loads(line)
                seen[row['id']] = row
    except FileNotFoundError:
        pass
    return seen


def _write_row(source: str, row: dict) -> None:
    with _LOCK:
        with open(_out_path(source), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def _extract_and_save(source: str, doc_id: int, text: str) -> dict:
    doc_type = 'cv' if source == 'resumes' else 'jd'
    try:
        view = extraction.extract_views(text, doc_type=doc_type)
        row = {
            'id': doc_id,
            'source': source,
            'ok': True,
            'error': None,
            'views': view,
        }
    except Exception as exc:
        row = {
            'id': doc_id,
            'source': source,
            'ok': False,
            'error': repr(exc),
            'views': None,
        }
    _write_row(source, row)
    return row


def extract_docs(
    df: pd.DataFrame,
    source: str,
    limit: int | None = None,
    workers: int = 3,
    resume: bool = True,
) -> dict:
    done = _load_done(source) if resume else {}
    ids = list(df.index)
    texts = df['text'].tolist()
    pending = [
        (int(doc_id), text)
        for doc_id, text in zip(ids, texts)
        if str(doc_id) not in done
    ]
    if limit is not None:
        pending = pending[:limit]

    skipped = dict(src=source, total=len(ids), pending=len(pending))
    print(f'[{source}] total={skipped["total"]} pending={skipped["pending"]}')

    stats = {'ok': len(done), 'fail': 0, 'fields_null': {f: 0 for f in _FIELDS}, 'n': 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_and_save, source, doc_id, text): doc_id
            for doc_id, text in pending
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f'extract {source}'):
            row = fut.result()
            stats['fail' if not row['ok'] else 'ok'] += 1
            if row['ok']:
                stats['n'] += 1
                for field in _FIELDS:
                    if not row['views'].get(field):
                        stats['fields_null'][field] += 1
    null_rates = {
        k: round(v / stats['n'], 3) if stats['n'] else 0
        for k, v in stats['fields_null'].items()
    }
    print(
        f'[{source}] ok={stats["ok"]} fail={stats["fail"]} null_rates={null_rates}'
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract semantic views via Qwen.')
    parser.add_argument(
        '--source',
        choices=['jobs', 'resumes', 'both'],
        default='both',
    )
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Re-extract everything, ignoring existing checkpoints.',
    )
    args = parser.parse_args()

    resume = not args.no_resume
    sources = ['jobs', 'resumes'] if args.source == 'both' else [args.source]
    for source in sources:
        df = (
            datasets.load_cleaned_jobs()
            if source == 'jobs'
            else datasets.load_resumes()
        )
        extract_docs(df, source, limit=args.limit, workers=args.workers, resume=resume)


if __name__ == '__main__':
    main()