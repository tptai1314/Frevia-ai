import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


def normalize_string_list(value: Any) -> list[str]:
    """Keep only usable string values returned for list fields by the LLM."""
    if value is None:
        return []
    if not isinstance(value, list):
        return value
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, dict):
            parts = [
                str(item[key]).strip()
                for key in ('degree', 'institution', 'field', 'years')
                if item.get(key) and str(item[key]).strip()
            ]
            if parts:
                result.append(' - '.join(parts))
    return result


def coerce_proficiency_level(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(1, min(10, int(value)))
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return max(1, min(10, int(match.group())))
    return None


def coerce_years(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r'\d+(\.\d+)?', value)
        if match:
            return float(match.group())
    return None


class SkillView(BaseModel):
    name: str
    proficiency_level: int | None = None

    @field_validator('proficiency_level', mode='before')
    @classmethod
    def validate_proficiency_level(cls, value: Any) -> int | None:
        return coerce_proficiency_level(value)


class ExperienceView(BaseModel):
    company: str | None = None
    position: str | None = None
    description: str | None = None
    years: float | None = None

    @field_validator('years', mode='before')
    @classmethod
    def validate_years(cls, value: Any) -> float | None:
        return coerce_years(value)


class ProjectView(BaseModel):
    name: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)

    @field_validator('technologies', mode='before')
    @classmethod
    def normalize_technologies(cls, value: Any) -> list[str]:
        return normalize_string_list(value)


class CVAnalysisResponse(BaseModel):
    professional_title: str | None = None
    summary: str | None = None
    skills: list[SkillView] = Field(default_factory=list)
    experience: list[ExperienceView] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    projects: list[ProjectView] = Field(default_factory=list)

    @field_validator('education', 'certifications', 'languages', mode='before')
    @classmethod
    def normalize_string_fields(cls, value: Any) -> list[str]:
        return normalize_string_list(value)