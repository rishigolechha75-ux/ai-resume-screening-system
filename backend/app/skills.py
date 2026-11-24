"""Skill taxonomy and dictionary-based skill extraction.

Each canonical skill maps to the aliases that may appear in a resume or job
description. Matching is done on cleaned text with word boundaries, so
"java" does not match inside "javascript".
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional, Pattern, Tuple

from .preprocessing import clean_text

SKILL_TAXONOMY: Dict[str, List[str]] = {
    # Programming languages
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp"],
    "Go": ["golang"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "Dart": ["dart"],
    "SQL": ["sql"],
    "R": ["r programming", "rstudio"],
    "Bash": ["bash", "shell scripting"],
    # Web
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express": ["express", "express.js"],
    "Redux": ["redux"],
    "Tailwind CSS": ["tailwind", "tailwind css"],
    "REST APIs": ["rest api", "rest apis", "restful", "restful apis"],
    "GraphQL": ["graphql"],
    # Backend / frameworks
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring Boot": ["spring boot", "spring"],
    "Microservices": ["microservices", "microservice"],
    # Databases
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Kafka": ["kafka"],
    # Data / ML
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "opencv"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow", "keras"],
    "PyTorch": ["pytorch"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Statistics": ["statistics", "statistical analysis", "hypothesis testing"],
    "Data Visualization": ["data visualization", "matplotlib", "seaborn"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Spark": ["spark", "pyspark"],
    "XGBoost": ["xgboost"],
    "LLMs": ["llm", "llms", "large language models", "langchain"],
    # Cloud / DevOps
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Terraform": ["terraform"],
    "Ansible": ["ansible"],
    "Jenkins": ["jenkins"],
    "CI/CD": ["ci/cd", "ci cd", "continuous integration", "github actions"],
    "Linux": ["linux", "unix"],
    "Prometheus": ["prometheus"],
    "Grafana": ["grafana"],
    "Git": ["git", "github", "gitlab"],
    # Mobile
    "Android": ["android"],
    "iOS": ["ios"],
    "Flutter": ["flutter"],
    "React Native": ["react native"],
    "Firebase": ["firebase"],
    # Testing
    "Unit Testing": ["unit testing", "pytest", "jest", "junit"],
    "Selenium": ["selenium"],
    # Business: HR
    "Recruitment": ["recruitment", "recruiting", "talent acquisition", "sourcing"],
    "Onboarding": ["onboarding"],
    "Payroll": ["payroll"],
    "Employee Relations": ["employee relations", "employee engagement"],
    "HRIS": ["hris", "workday", "bamboohr"],
    "Performance Management": ["performance management", "performance appraisal"],
    # Business: sales & marketing
    "SEO": ["seo", "search engine optimization"],
    "Digital Marketing": ["digital marketing", "social media marketing", "google ads"],
    "Content Marketing": ["content marketing", "copywriting"],
    "CRM": ["crm", "salesforce", "hubspot"],
    "Lead Generation": ["lead generation", "prospecting"],
    "Negotiation": ["negotiation"],
    "Market Research": ["market research"],
    # Business: finance
    "Financial Analysis": ["financial analysis", "financial modeling", "financial modelling"],
    "Accounting": ["accounting", "bookkeeping", "gaap", "ifrs"],
    "Excel": ["excel", "vlookup", "pivot tables"],
    "Budgeting": ["budgeting", "forecasting", "variance analysis"],
    "Taxation": ["taxation", "tax", "gst"],
    "Auditing": ["auditing", "audit"],
    "Tally": ["tally"],
    # Soft skills
    "Communication": ["communication", "communication skills"],
    "Leadership": ["leadership", "team lead", "mentoring"],
    "Agile": ["agile", "scrum", "jira"],
}


@lru_cache(maxsize=1)
def _compiled_patterns() -> List[Tuple[str, Pattern[str]]]:
    patterns = []
    for skill, aliases in SKILL_TAXONOMY.items():
        # longest aliases first so "react native" wins over "react"
        alternation = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
        patterns.append((skill, re.compile(rf"(?<![a-z0-9+#])(?:{alternation})(?![a-z0-9+#])")))
    return patterns


def extract_skills(text: str) -> List[str]:
    """Return the canonical skills mentioned in ``text``, sorted alphabetically."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    found = {skill for skill, pattern in _compiled_patterns() if pattern.search(cleaned)}
    # "React Native" mentions should not also count as plain "React"
    if "React Native" in found and not re.search(
        r"(?<![a-z0-9])react(?:\.js|js)?(?!\s+native)(?![a-z0-9])", cleaned
    ):
        found.discard("React")
    return sorted(found, key=str.lower)


_OPTIONAL_HEADING = re.compile(
    r"^\s*[-*]?\s*(nice[\s-]to[\s-]have|preferred|good[\s-]to[\s-]have|bonus|optional)\b",
    re.IGNORECASE | re.MULTILINE,
)

NICE_TO_HAVE_WEIGHT = 0.5


def extract_job_skills(job_description: str) -> Tuple[List[str], List[str]]:
    """Split a job description's skills into ``(required, nice_to_have)``.

    Everything after a heading such as "Nice to have:" or "Preferred:" counts
    as optional; all other skills are treated as required.
    """
    match = _OPTIONAL_HEADING.search(job_description)
    if not match:
        return extract_skills(job_description), []
    required = extract_skills(job_description[: match.start()])
    optional = [s for s in extract_skills(job_description[match.start():]) if s not in required]
    return required, optional


def skill_match(
    required: List[str], resume_skills: List[str], nice_to_have: Optional[List[str]] = None
) -> Tuple[List[str], List[str], float]:
    """Compare a job's skills with a resume's skills.

    Returns ``(matched, missing, coverage)``. Coverage is a weighted share of
    the job's skills found in the resume, where nice-to-have skills count half
    as much as required ones. It is 1.0 when the job lists no skills.
    """
    nice_to_have = nice_to_have or []
    resume_set = set(resume_skills)
    job_skills = required + nice_to_have
    matched = [s for s in job_skills if s in resume_set]
    missing = [s for s in job_skills if s not in resume_set]
    total = len(required) + NICE_TO_HAVE_WEIGHT * len(nice_to_have)
    if not total:
        return matched, missing, 1.0
    got = sum(1 for s in required if s in resume_set) + NICE_TO_HAVE_WEIGHT * sum(
        1 for s in nice_to_have if s in resume_set
    )
    return matched, missing, got / total
