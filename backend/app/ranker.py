"""Rank resumes against a job description.

Each resume gets four component scores in the range 0-1, combined into a
weighted final score (0-100):

=====================  ======  ==================================================
Component              Weight  How it is computed
=====================  ======  ==================================================
Content similarity     0.40    TF-IDF cosine similarity between JD and resume
Skill coverage         0.35    Share of the JD's skills found in the resume
                               (nice-to-have skills count half)
Category fit           0.15    Classifier probability that the resume belongs to
                               the JD's predicted job category
Experience fit         0.10    Candidate years / required years (capped at 1)
=====================  ======  ==================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .classifier import CategoryClassifier
from .preprocessing import clean_text, extract_years_of_experience
from .skills import extract_job_skills, extract_skills, skill_match

DEFAULT_WEIGHTS: Dict[str, float] = {
    "similarity": 0.40,
    "skills": 0.35,
    "category": 0.15,
    "experience": 0.10,
}

# Raw TF-IDF cosine between a JD and a resume rarely exceeds ~0.5 even for a
# strong match, so the similarity is rescaled: a cosine of 0.5 or more maps to 1.0.
SIMILARITY_CEILING = 0.5

SHORTLIST_THRESHOLD = 60.0
REVIEW_THRESHOLD = 40.0


@dataclass
class ResumeInput:
    name: str
    text: str


@dataclass
class JobProfile:
    skills: List[str]
    nice_to_have_skills: List[str]
    required_years: Optional[float]
    category: Optional[str]
    category_confidence: Optional[float]


@dataclass
class RankedResume:
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
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyse_job(job_description: str, classifier: Optional[CategoryClassifier] = None) -> JobProfile:
    """Extract the skills, experience requirement and category of a job description."""
    category, confidence = (None, None)
    if classifier is not None and classifier.is_trained:
        category, confidence = classifier.predict(job_description)
        confidence = round(confidence, 4)
    required, nice_to_have = extract_job_skills(job_description)
    return JobProfile(
        skills=required,
        nice_to_have_skills=nice_to_have,
        required_years=extract_years_of_experience(job_description),
        category=category,
        category_confidence=confidence,
    )


def recommendation_for(score: float) -> str:
    if score >= SHORTLIST_THRESHOLD:
        return "Shortlist"
    if score >= REVIEW_THRESHOLD:
        return "Review"
    return "Reject"


def tfidf_similarities(job_description: str, resume_texts: List[str]) -> List[float]:
    """Cosine similarity between the JD and each resume in a shared TF-IDF space."""
    corpus = [job_description] + resume_texts
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()


def rank_resumes(
    job_description: str,
    resumes: List[ResumeInput],
    classifier: Optional[CategoryClassifier] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Score and rank resumes for a job description.

    Returns a dict with the analysed ``job`` profile and the ranked ``results``.
    """
    if not job_description.strip():
        raise ValueError("Job description is empty.")
    if not resumes:
        raise ValueError("No resumes were provided.")

    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_weight = sum(weights.values())
    job = analyse_job(job_description, classifier)
    similarities = tfidf_similarities(job_description, [r.text for r in resumes])

    results: List[RankedResume] = []
    for resume, raw_similarity in zip(resumes, similarities):
        resume_skills = extract_skills(resume.text)
        matched, missing, coverage = skill_match(job.skills, resume_skills, job.nice_to_have_skills)
        similarity = min(1.0, raw_similarity / SIMILARITY_CEILING)

        predicted, confidence, category_fit = None, None, 0.0
        if classifier is not None and classifier.is_trained:
            probs = classifier.predict_proba(resume.text)
            predicted, confidence = next(iter(probs.items()))
            category_fit = probs.get(job.category, 0.0) if job.category else 0.0

        years = extract_years_of_experience(resume.text)
        if job.required_years:
            experience_fit = min(1.0, (years or 0.0) / job.required_years)
        else:
            experience_fit = 1.0

        score = 100.0 * (
            weights["similarity"] * similarity
            + weights["skills"] * coverage
            + weights["category"] * category_fit
            + weights["experience"] * experience_fit
        ) / total_weight

        results.append(
            RankedResume(
                rank=0,
                name=resume.name,
                score=round(score, 1),
                recommendation=recommendation_for(score),
                similarity=round(raw_similarity, 4),
                skill_coverage=round(coverage, 4),
                category_fit=round(category_fit, 4),
                experience_fit=round(experience_fit, 4),
                predicted_category=predicted,
                category_confidence=round(confidence, 4) if confidence is not None else None,
                years_of_experience=years,
                matched_skills=matched,
                missing_skills=missing,
                extra_skills=[s for s in resume_skills if s not in set(job.skills + job.nice_to_have_skills)],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    for position, result in enumerate(results, start=1):
        result.rank = position

    return {"job": asdict(job), "weights": weights, "results": [r.to_dict() for r in results]}
