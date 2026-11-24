"""Unit tests for text cleaning, skill extraction, classification and ranking."""

import pytest

from app.classifier import load_or_train
from app.preprocessing import clean_text, extract_years_of_experience
from app.ranker import ResumeInput, rank_resumes, recommendation_for
from app.skills import extract_job_skills, extract_skills, skill_match


@pytest.fixture(scope="module")
def classifier():
    return load_or_train()


# --- preprocessing -----------------------------------------------------------

def test_clean_text_removes_contact_details_and_keeps_tech_terms():
    text = "Email: jane@example.com, Phone: +91 98765 43210. Skills: C++, C#, Node.js! https://x.io"
    cleaned = clean_text(text)
    assert "example.com" not in cleaned
    assert "98765" not in cleaned
    assert "https" not in cleaned
    for term in ("c++", "c#", "node.js"):
        assert term in cleaned


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Backend developer with 4 years of experience in Python", 4.0),
        ("Experience of 2.5 years in recruitment", 2.5),
        ("We need 3+ years of professional experience", 3.0),
        ("Worked 2 years at A and 6 years at B", 6.0),
        ("Fresh graduate", None),
        ("", None),
    ],
)
def test_extract_years_of_experience(text, expected):
    assert extract_years_of_experience(text) == expected


# --- skills ------------------------------------------------------------------

def test_extract_skills_uses_word_boundaries():
    skills = extract_skills("Strong JavaScript and TypeScript; built REST APIs with Node.js")
    assert "JavaScript" in skills and "TypeScript" in skills
    assert "Java" not in skills  # 'java' must not match inside 'javascript'
    assert "Node.js" in skills and "REST APIs" in skills


def test_extract_skills_maps_aliases_to_canonical_names():
    skills = extract_skills("k8s, sklearn, postgres, pytest and github actions")
    assert {"Kubernetes", "scikit-learn", "PostgreSQL", "Unit Testing", "CI/CD"} <= set(skills)


def test_react_native_does_not_imply_react():
    assert "React" not in extract_skills("Mobile apps in React Native")
    assert {"React", "React Native"} <= set(extract_skills("React web apps and React Native mobile apps"))


def test_job_skills_split_required_and_nice_to_have():
    jd = "Required: Python, Django, PostgreSQL\nNice to have: Kafka, Docker, Python"
    required, nice = extract_job_skills(jd)
    assert set(required) == {"Python", "Django", "PostgreSQL"}
    assert set(nice) == {"Kafka", "Docker"}  # Python is already required


def test_skill_match_weights_nice_to_have_at_half():
    matched, missing, coverage = skill_match(["Python", "SQL"], ["Python", "Docker"], ["Docker", "Kafka"])
    assert matched == ["Python", "Docker"]
    assert missing == ["SQL", "Kafka"]
    # (1 required + 0.5 * 1 optional) / (2 + 0.5 * 2) = 1.5 / 3
    assert coverage == pytest.approx(0.5)


def test_skill_match_with_no_job_skills_is_full_coverage():
    assert skill_match([], ["Python"]) == ([], [], 1.0)


# --- classifier --------------------------------------------------------------

def test_classifier_predicts_obvious_categories(classifier):
    hr_text = "HR executive handling recruitment, onboarding, payroll and employee relations in Workday"
    devops_text = "DevOps engineer automating AWS with Terraform, Kubernetes, Jenkins CI/CD and Prometheus"
    assert classifier.predict(hr_text)[0] == "Human Resources"
    assert classifier.predict(devops_text)[0] == "DevOps Engineering"


def test_classifier_probabilities_sum_to_one(classifier):
    probabilities = classifier.predict_proba("python developer")
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert list(probabilities.values()) == sorted(probabilities.values(), reverse=True)


# --- ranker ------------------------------------------------------------------

JD = (
    "Backend Python Developer with 3+ years of experience.\n"
    "Required skills: Python, FastAPI, PostgreSQL, Docker, REST APIs\n"
    "Nice to have: Kafka"
)
STRONG = "Backend developer, 4 years of experience. Python, FastAPI, PostgreSQL, Docker, Kafka, REST APIs."
WEAK = "HR executive with 5 years of experience in recruitment, payroll and onboarding."


def test_ranker_puts_matching_resume_first(classifier):
    result = rank_resumes(JD, [ResumeInput("weak", WEAK), ResumeInput("strong", STRONG)], classifier)
    names = [r["name"] for r in result["results"]]
    assert names == ["strong", "weak"]
    top = result["results"][0]
    assert top["rank"] == 1
    assert top["skill_coverage"] == pytest.approx(1.0)
    assert top["missing_skills"] == []
    assert result["job"]["required_years"] == 3.0
    assert result["job"]["nice_to_have_skills"] == ["Kafka"]


def test_ranker_scores_are_bounded(classifier):
    result = rank_resumes(JD, [ResumeInput("strong", STRONG), ResumeInput("weak", WEAK)], classifier)
    for r in result["results"]:
        assert 0 <= r["score"] <= 100


def test_ranker_works_without_classifier():
    result = rank_resumes(JD, [ResumeInput("strong", STRONG)], classifier=None)
    assert result["results"][0]["predicted_category"] is None
    assert result["results"][0]["category_fit"] == 0.0


def test_ranker_custom_weights_tfidf_only(classifier):
    weights = {"similarity": 1, "skills": 0, "category": 0, "experience": 0}
    result = rank_resumes(JD, [ResumeInput("strong", STRONG)], classifier, weights)
    r = result["results"][0]
    assert r["score"] == pytest.approx(100 * min(1.0, r["similarity"] / 0.5), abs=0.1)


@pytest.mark.parametrize("jd, resumes", [("   ", [ResumeInput("a", "text")]), (JD, [])])
def test_ranker_rejects_empty_input(jd, resumes):
    with pytest.raises(ValueError):
        rank_resumes(jd, resumes)


@pytest.mark.parametrize("score, label", [(75, "Shortlist"), (60, "Shortlist"), (45, "Review"), (10, "Reject")])
def test_recommendation_thresholds(score, label):
    assert recommendation_for(score) == label
