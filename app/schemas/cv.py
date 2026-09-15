from typing import Any

from pydantic import BaseModel, Field, field_validator


def normalize_string_list(value: Any) -> list[str]:
    """Keep only usable string values returned for list fields by the LLM."""
    if value is None:
        return []
    if not isinstance(value, list):
        return value
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


class SkillView(BaseModel):
    name: str
    level: str | None = None


class ExperienceView(BaseModel):
    company: str | None = None
    position: str | None = None
    description: str | None = None
    years: float | None = None


class EducationView(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field: str | None = None
    years: str | None = None


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
    education: list[EducationView] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    projects: list[ProjectView] = Field(default_factory=list)

    @field_validator('certifications', 'languages', mode='before')
    @classmethod
    def normalize_string_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)
