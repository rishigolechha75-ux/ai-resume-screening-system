"""Request and response models for the REST API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ResumeText(BaseModel):
    name: str = Field(..., examples=["jane_doe.pdf"])
    text: str = Field(..., min_length=1)


class ScreenTextRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    resumes: List[ResumeText] = Field(..., min_length=1)
    weights: Optional[Dict[str, float]] = Field(
        None, description="Optional override of similarity/skills/category/experience weights"
    )


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class JobProfileOut(BaseModel):
    skills: List[str]
    nice_to_have_skills: List[str]
    required_years: Optional[float]
    category: Optional[str]
    category_confidence: Optional[float]


class RankedResumeOut(BaseModel):
    rank: int
    name: str
    score: float
    recommendation: str
    similarity: float
    skill_coverage: float
    category_fit: float
    experience_fit: float
    predicted_category: Optional[str]
    category_confidence: Optional[float]
    years_of_experience: Optional[float]
    matched_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]


class FileError(BaseModel):
    name: str
    error: str


class ScreenResponse(BaseModel):
    job: JobProfileOut
    weights: Dict[str, float]
    results: List[RankedResumeOut]
    errors: List[FileError] = []


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    probabilities: Dict[str, float]


class SkillsResponse(BaseModel):
    skills: List[str]
    years_of_experience: Optional[float]
