from pydantic import BaseModel, Field


class SkillView(BaseModel):
    name: str
    level: str | None = None


class ExperienceView(BaseModel):
    company: str | None = None
    position: str | None = None
    description: str | None = None
    years: float | None = None


class CVAnalysisResponse(BaseModel):
    professional_title: str | None = None
    summary: str | None = None
    skills: list[SkillView] = Field(default_factory=list)
    experience: list[ExperienceView] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)