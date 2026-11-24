"""FastAPI application exposing the resume screening pipeline as a REST API.

Run locally:
    uvicorn app.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .classifier import load_or_train
from .parsing import SUPPORTED_EXTENSIONS, UnsupportedFileError, extract_text
from .preprocessing import extract_years_of_experience
from .ranker import DEFAULT_WEIGHTS, ResumeInput, rank_resumes
from .schemas import (
    ClassifyResponse,
    ScreenResponse,
    ScreenTextRequest,
    SkillsResponse,
    TextRequest,
)
from .skills import extract_skills

MAX_FILES = 50
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB per resume
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"

app = FastAPI(
    title="AI Resume Screening System",
    description="Rank resumes against a job description using TF-IDF similarity, "
    "skill matching and a machine-learning category classifier.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = load_or_train()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model_loaded": classifier.is_trained}


@app.get("/api/config")
def config() -> dict:
    return {
        "categories": classifier.categories,
        "default_weights": DEFAULT_WEIGHTS,
        "supported_file_types": sorted(SUPPORTED_EXTENSIONS),
        "max_files": MAX_FILES,
    }


@app.post("/api/screen", response_model=ScreenResponse)
async def screen_files(
    job_description: str = Form(...),
    files: List[UploadFile] = File(...),
    weights: Optional[str] = Form(None, description='Optional JSON, e.g. {"skills": 0.5}'),
) -> dict:
    """Upload resume files (PDF, DOCX, TXT) and rank them against a job description."""
    parsed_weights = None
    if weights:
        try:
            parsed_weights = {k: float(v) for k, v in json.loads(weights).items()}
        except (ValueError, AttributeError, TypeError):
            raise HTTPException(422, "weights must be a JSON object of numbers.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Upload at most {MAX_FILES} resumes at a time.")

    resumes, errors = [], []
    for upload in files:
        name = upload.filename or "resume"
        content = await upload.read()
        if len(content) > MAX_FILE_BYTES:
            errors.append({"name": name, "error": "File is larger than 5 MB."})
            continue
        try:
            text = extract_text(name, content)
        except UnsupportedFileError as exc:
            errors.append({"name": name, "error": str(exc)})
            continue
        except Exception:  # corrupt or encrypted file
            errors.append({"name": name, "error": "Could not read this file."})
            continue
        if not text.strip():
            errors.append({"name": name, "error": "No text found (scanned image PDFs are not supported)."})
            continue
        resumes.append(ResumeInput(name=name, text=text))

    if not resumes:
        raise HTTPException(400, {"message": "None of the files could be read.", "errors": errors})

    result = _rank(job_description, resumes, parsed_weights)
    result["errors"] = errors
    return result


@app.post("/api/screen/text", response_model=ScreenResponse)
def screen_text(request: ScreenTextRequest) -> dict:
    """Rank resumes supplied as plain text (useful for integrations and testing)."""
    resumes = [ResumeInput(name=r.name, text=r.text) for r in request.resumes]
    return _rank(request.job_description, resumes, request.weights)


@app.get("/api/samples")
def samples() -> dict:
    """Return the bundled sample job description and resumes for a quick demo."""
    if not SAMPLES_DIR.exists():
        raise HTTPException(404, "Sample files are not available.")
    return {
        "job_description": (SAMPLES_DIR / "job_description.txt").read_text(),
        "resumes": [
            {"name": path.name, "text": path.read_text()}
            for path in sorted((SAMPLES_DIR / "resumes").glob("*.txt"))
        ],
    }


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(request: TextRequest) -> dict:
    """Predict the job category of a resume or job description."""
    probabilities = classifier.predict_proba(request.text)
    category, confidence = next(iter(probabilities.items()))
    return {"category": category, "confidence": confidence, "probabilities": probabilities}


@app.post("/api/skills", response_model=SkillsResponse)
def skills(request: TextRequest) -> dict:
    """Extract skills and years of experience from any text."""
    return {
        "skills": extract_skills(request.text),
        "years_of_experience": extract_years_of_experience(request.text),
    }


def _rank(job_description: str, resumes: List[ResumeInput], weights=None) -> dict:
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise HTTPException(422, f"Unknown weight keys: {sorted(unknown)}")
        if any(w < 0 for w in weights.values()) or sum({**DEFAULT_WEIGHTS, **weights}.values()) <= 0:
            raise HTTPException(422, "Weights must be non-negative and not all zero.")
    try:
        return rank_resumes(job_description, resumes, classifier, weights)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
