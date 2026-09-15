import json

import httpx

from app.schemas.cv import CVAnalysisResponse


OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'qwen3:4b'
OLLAMA_TIMEOUT = 300.0


CV_ANALYSIS_PROMPT = """
You are an AI system for extracting structured information from resumes/CVs.

Return ONLY valid JSON. Do not use Markdown or explanations.

The JSON must contain EXACTLY these fields:

{
  "professional_title": string or null,
  "summary": string or null,z
  "skills": [
    {
      "name": string,
      "level": string or null
    }
  ],
  "experience": [
    {
      "company": string or null,
      "position": string or null,
      "description": string or null,
      "years": number or null
    }
  ],
  "education": [
    {
      "institution": string or null,
      "degree": string or null,
      "field": string or null,
      "years": string or null
    }
  ],
  "certifications": [string],
  "languages": [string],
  "projects": [
    {
      "name": string or null,
      "description": string or null,
      "technologies": [string]
    }
  ]
}

RULES:

1. Extract information ONLY from the CV text.
2. Never invent, infer, or fabricate information.
3. If a single value is missing or unclear, use null.
4. If a list has no valid items, use [].
5. NEVER put null inside an array.
6. Preserve the meaning of the original CV.
7. Skills, technologies, certifications, languages, companies, education,
   projects, and job positions must be explicitly supported by the CV.
8. Do not convert skills into certifications.
9. Do not infer languages from the language used in the CV.
10. For experience years, calculate only when the dates are explicitly available.
11. CV text may contain spacing, OCR, or character-extraction errors.
    Correct only obvious extraction noise; never guess missing information.
12. When uncertain, prefer null or [] rather than guessing.

CV TEXT:
"""


async def analyze_cv_with_qwen(cv_text: str) -> CVAnalysisResponse:
    prompt = f'{CV_ANALYSIS_PROMPT}\n{cv_text}'

    payload = {
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'format': 'json',
        'think': False,
    }

    try:
        async with httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT,
        ) as client:
            response = await client.post(
                OLLAMA_URL,
                json=payload,
            )

        response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise ValueError(
            'Ollama request timed out while analyzing the CV.'
        ) from exc

    except httpx.HTTPError as exc:
        raise ValueError(
            f'Could not connect to Ollama: {exc!r}'
        ) from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise ValueError(
            'Ollama returned an invalid HTTP response.'
        ) from exc

    response_text = result.get('response')

    if not response_text:
        raise ValueError(
            'Ollama returned an empty response.'
        )

    try:
        parsed_result = json.loads(response_text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            'Qwen returned invalid JSON.'
        ) from exc

    try:
        return CVAnalysisResponse.model_validate(parsed_result)

    except Exception as exc:
        raise ValueError(
            f'Qwen returned JSON that does not match the CV schema: {exc}'
        ) from exc
