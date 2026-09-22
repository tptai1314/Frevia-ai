"""Qwen semantic view extraction for FREVIA.

Extracts the 4 semantic views (title, skills, experience, projects) from raw
CV or JD text into a fixed JSON structure. Uses a local Ollama instance.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'qwen3:4b'
OLLAMA_TIMEOUT = 300.0
OLLAMA_NUM_CTX = 4096
DEFAULT_MAX_CHARS = 3500

VIEW_SCHEMA = {
    'title': 'string or null',
    'skills': ['string'],
    'experience': [
        {
            'position': 'string or null',
            'description': 'string or null',
            'years': 'number or null',
        }
    ],
    'projects': [
        {
            'name': 'string or null',
            'description': 'string or null',
            'technologies': ['string'],
        }
    ],
}

_SHARED_RULES = """
Return ONLY valid JSON. Do not use Markdown or explanations.
The JSON must contain EXACTLY these fields:

{
  "title": string or null,
  "skills": [string],
  "experience": [
    {
      "position": string or null,
      "description": string or null,
      "years": number or null
    }
  ],
  "projects": [
    {
      "name": string or null,
      "description": string or null,
      "technologies": [string]
    }
  ]
}

RULES:
1. Extract information ONLY from the given text. Never invent or infer.
2. If a single value is missing or unclear, use null.
3. If a list has no valid items, use [] and never put null inside an array.
4. skills: one item per distinct skill/technology.
5. experience: one item per distinct role/position or period described.
   Compute years only when dates are explicitly available; otherwise null.
6. projects: one item per distinct project/initiative mentioned.
7. Text may contain OCR/extraction noise. Correct only obvious noise;
   never guess missing information.
8. When uncertain, prefer null or [] rather than guessing.
"""

CV_PROMPT = (
    'You are an AI extracting structured information from a RESUME/CV.\n'
    + _SHARED_RULES
    + '\nRESUME TEXT:\n'
)

JD_PROMPT = (
    'You are an AI extracting structured information from a JOB DESCRIPTION.\n'
    'Interpret each field as the REQUIREMENT for the role:\n'
    '- title: the job title.\n'
    '- skills: required skills/technologies.\n'
    '- experience: required/expected experience.\n'
    '- projects: projects, products, or use-cases mentioned.\n'
    + _SHARED_RULES
    + '\nJOB DESCRIPTION TEXT:\n'
)


def _call_ollama(prompt: str) -> dict[str, Any]:
    payload = {
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'format': 'json',
        'think': False,
        'num_ctx': OLLAMA_NUM_CTX,
    }
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        response = client.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    text = result.get('response', '')
    if not text:
        raise ValueError('Ollama returned an empty response')
    return json.loads(text)


def extract_views(
    text: str,
    doc_type: str = 'cv',
    retries: int = 1,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Extract semantic views with a single retry on malformed JSON."""
    prompt = CV_PROMPT if doc_type == 'cv' else JD_PROMPT
    full_prompt = f'{prompt.rstrip()}\n{text[:max_chars]}'
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            parsed = _call_ollama(full_prompt)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError('Expected a JSON object')
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
    raise ValueError(f'Extraction failed: {last_error}') from last_error


def serialize_view(view: dict[str, Any]) -> str:
    """Normalize a parsed view dict into a flat text for downstream matching."""
    parts: list[str] = []
    if view.get('title'):
        parts.append(str(view['title']))
    skills = [s for s in view.get('skills', []) if isinstance(s, str)]
    if skills:
        parts.append('skills: ' + ' '.join(skills))
    for exp in view.get('experience', []) or []:
        if not isinstance(exp, dict):
            continue
        seg = ' '.join(
            str(exp[k]) for k in ('position', 'description') if exp.get(k)
        )
        if seg:
            parts.append(f'experience: {seg}')
    for proj in view.get('projects', []) or []:
        if not isinstance(proj, dict):
            continue
        seg = ' '.join(
            str(proj[k])
            for k in ('name', 'description')
            if proj.get(k)
        )
        techs = [t for t in proj.get('technologies', []) if isinstance(t, str)]
        if techs:
            seg += ' ' + ' '.join(techs)
        if seg:
            parts.append(f'project: {seg}')
    return ' '.join(parts)