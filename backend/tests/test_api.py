"""End-to-end tests for the REST API using FastAPI's TestClient."""

import io

import docx
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

JD = "Python backend developer, 2+ years of experience. Required: Python, Django, PostgreSQL, Docker."
STRONG = "Software engineer with 3 years of experience in Python, Django, PostgreSQL and Docker."
WEAK = "Sales executive with 4 years of experience in lead generation, CRM and negotiation."


def _docx_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    document = docx.Document()
    for line in text.split(". "):
        document.add_paragraph(line)
    document.save(buffer)
    return buffer.getvalue()


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_config_lists_categories_and_file_types():
    body = client.get("/api/config").json()
    assert len(body["categories"]) == 8
    assert body["supported_file_types"] == [".docx", ".pdf", ".txt"]


def test_screen_text_ranks_resumes():
    response = client.post(
        "/api/screen/text",
        json={"job_description": JD, "resumes": [{"name": "weak", "text": WEAK}, {"name": "strong", "text": STRONG}]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["name"] for r in results] == ["strong", "weak"]
    assert results[0]["recommendation"] in {"Shortlist", "Review"}


def test_screen_text_rejects_unknown_weight_keys():
    response = client.post(
        "/api/screen/text",
        json={"job_description": JD, "resumes": [{"name": "a", "text": STRONG}], "weights": {"luck": 1}},
    )
    assert response.status_code == 422


def test_screen_text_validates_empty_resume_list():
    response = client.post("/api/screen/text", json={"job_description": JD, "resumes": []})
    assert response.status_code == 422


def test_screen_files_accepts_txt_and_docx_and_reports_bad_files():
    files = [
        ("files", ("strong.docx", _docx_bytes(STRONG), "application/octet-stream")),
        ("files", ("weak.txt", WEAK.encode(), "text/plain")),
        ("files", ("photo.png", b"\x89PNG", "image/png")),
    ]
    response = client.post("/api/screen", data={"job_description": JD}, files=files)
    assert response.status_code == 200
    body = response.json()
    assert [r["name"] for r in body["results"]] == ["strong.docx", "weak.txt"]
    assert body["errors"] == [
        {"name": "photo.png", "error": "Unsupported file type '.png'. Use one of: .docx, .pdf, .txt"}
    ]


def test_screen_files_fails_when_nothing_is_readable():
    files = [("files", ("empty.txt", b"   ", "text/plain"))]
    response = client.post("/api/screen", data={"job_description": JD}, files=files)
    assert response.status_code == 400


@pytest.mark.parametrize("endpoint", ["/api/classify", "/api/skills"])
def test_text_endpoints_require_text(endpoint):
    assert client.post(endpoint, json={"text": ""}).status_code == 422


def test_classify_endpoint():
    body = client.post("/api/classify", json={"text": WEAK}).json()
    assert body["category"] == "Sales & Marketing"
    assert 0 < body["confidence"] <= 1


def test_skills_endpoint():
    body = client.post("/api/skills", json={"text": STRONG}).json()
    assert {"Python", "Django", "PostgreSQL", "Docker"} <= set(body["skills"])
    assert body["years_of_experience"] == 3.0


def test_samples_endpoint_returns_demo_data():
    body = client.get("/api/samples").json()
    assert "Backend Python Developer" in body["job_description"]
    assert len(body["resumes"]) == 5


def test_screen_files_accepts_weights_json():
    files = [("files", ("strong.txt", STRONG.encode(), "text/plain"))]
    weights = '{"similarity": 0, "skills": 1, "category": 0, "experience": 0}'
    body = client.post("/api/screen", data={"job_description": JD, "weights": weights}, files=files).json()
    assert body["results"][0]["score"] == 100.0


def test_screen_files_rejects_malformed_weights():
    files = [("files", ("strong.txt", STRONG.encode(), "text/plain"))]
    response = client.post("/api/screen", data={"job_description": JD, "weights": "not json"}, files=files)
    assert response.status_code == 422
