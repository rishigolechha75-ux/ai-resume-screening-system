"""Generate the bundled synthetic resume dataset and matching job descriptions.

Why synthetic? Public resume datasets often carry licensing or privacy
restrictions, so the repository ships a reproducible, fully synthetic dataset
instead. Every name and company below is fictional. To make the task realistic,
resumes blend in content from neighbouring roles (a backend resume may read
partly like a frontend or DevOps one), use generic titles such as "Software
Engineer", and share generic duties across roles.

To train on a real dataset instead, see ``scripts/train.py --data``.

Usage:
    python scripts/generate_dataset.py            # writes data/resumes.csv and data/job_descriptions.csv
    python scripts/generate_dataset.py --per-category 80 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FIRST_NAMES = [
    "Aarav", "Ananya", "Vihaan", "Diya", "Arjun", "Isha", "Kabir", "Meera", "Rohan", "Saanvi",
    "Aditya", "Priya", "Neel", "Tara", "Kiran", "Riya", "Dev", "Nisha", "Yash", "Pooja",
    "Liam", "Emma", "Noah", "Olivia", "Ethan", "Sofia", "Lucas", "Mia", "Omar", "Zara",
]
LAST_NAMES = [
    "Sharma", "Iyer", "Mehta", "Kapoor", "Nair", "Rao", "Gupta", "Joshi", "Desai", "Reddy",
    "Bose", "Patel", "Khan", "Fernandes", "Menon", "Smith", "Garcia", "Chen", "Silva", "Ali",
]
COMPANIES = [
    "Northwind Labs", "BluePeak Systems", "Quantra Analytics", "Helix Softworks", "Orbit Retail",
    "Lumen Health", "Crestline Finance", "Nimbus Cloud Co", "Vertex Media", "Saffron Foods",
    "Ironleaf Logistics", "Pixelcraft Studio", "Greenfield Energy", "Atlas Insurance",
]
DEGREES = {
    "tech": ["B.Tech in Computer Engineering", "B.E. in Information Technology",
             "B.Sc in Computer Science", "M.Tech in Computer Science", "MCA"],
    "business": ["MBA", "BBA", "B.Com", "M.Com", "BA in Psychology", "MBA in Human Resources"],
}

# Each category: titles, core skills (sampled), duties (templated bullet points)
CATEGORIES = {
    "Data Science": {
        "degree": "tech",
        "titles": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "ML Research Intern"],
        "skills": ["Python", "machine learning", "scikit-learn", "Pandas", "NumPy", "SQL", "statistics",
                   "deep learning", "TensorFlow", "PyTorch", "NLP", "XGBoost", "data visualization",
                   "Tableau", "Spark", "computer vision", "hypothesis testing", "LLMs"],
        "duties": [
            "built a {s1} model to predict customer churn, improving recall by {n}%",
            "cleaned and analysed large datasets using {s1} and {s2}",
            "designed A/B experiments and applied {s1} to evaluate product changes",
            "deployed {s1} models as REST APIs and monitored model drift",
            "created dashboards in {s1} to track business KPIs for leadership",
            "trained {s1} models for text classification on {n}k labelled documents",
            "engineered features from transactional data and tuned {s1} hyperparameters",
        ],
    },
    "Frontend Development": {
        "degree": "tech",
        "titles": ["Frontend Developer", "UI Engineer", "React Developer", "Web Developer"],
        "skills": ["JavaScript", "TypeScript", "React", "Redux", "HTML", "CSS", "Tailwind CSS",
                   "Next.js", "Vue", "Angular", "Jest", "Figma", "responsive design", "web performance",
                   "accessibility", "REST APIs", "GraphQL", "Git"],
        "duties": [
            "built reusable {s1} components used across {n} product pages",
            "improved page load time by {n}% through code splitting and {s1} optimisation",
            "implemented responsive layouts with {s1} and {s2} from Figma designs",
            "integrated {s1} endpoints and managed client state with {s2}",
            "wrote unit tests with {s1} raising coverage to {n}%",
            "led migration of a legacy jQuery app to {s1}",
            "fixed accessibility issues to meet WCAG guidelines using {s1}",
        ],
    },
    "Backend Development": {
        "degree": "tech",
        "titles": ["Backend Developer", "Software Engineer", "Python Developer", "Java Developer"],
        "skills": ["Python", "Java", "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot",
                   "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "REST APIs", "microservices",
                   "GraphQL", "unit testing", "Docker", "Git", "Go"],
        "duties": [
            "designed {s1} services handling {n}k requests per minute",
            "built REST APIs with {s1} backed by {s2}",
            "reduced query latency by {n}% by adding {s1} caching and indexes",
            "implemented authentication and role-based authorization in {s1}",
            "split a monolith into {s1} communicating over {s2}",
            "wrote integration tests with {s1} and set up code reviews",
            "designed relational schemas in {s1} for billing and orders",
        ],
    },
    "DevOps Engineering": {
        "degree": "tech",
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Platform Engineer"],
        "skills": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins",
                   "CI/CD", "GitHub Actions", "Linux", "Bash", "Prometheus", "Grafana", "Python",
                   "Helm", "networking", "incident response"],
        "duties": [
            "automated infrastructure provisioning on {s1} using {s2}",
            "built {s1} pipelines that cut release time from days to {n} minutes",
            "migrated {n} services to {s1} clusters with zero downtime",
            "set up monitoring and alerting with {s1} and {s2}",
            "hardened {s1} servers and wrote {s2} scripts for routine operations",
            "reduced cloud spend by {n}% through rightsizing on {s1}",
            "led incident response and postmortems for production outages",
        ],
    },
    "Mobile Development": {
        "degree": "tech",
        "titles": ["Android Developer", "iOS Developer", "Flutter Developer", "Mobile Engineer"],
        "skills": ["Android", "Kotlin", "Java", "iOS", "Swift", "Flutter", "Dart", "React Native",
                   "Firebase", "REST APIs", "SQLite", "Jetpack Compose", "SwiftUI", "Git",
                   "push notifications", "app store deployment"],
        "duties": [
            "developed {s1} apps in {s2} with over {n}k downloads",
            "integrated {s1} for authentication, analytics and push notifications",
            "rebuilt the checkout flow in {s1}, lifting conversion by {n}%",
            "implemented offline sync with local storage and {s1} endpoints",
            "published releases to the Play Store and App Store using {s1}",
            "reduced app crash rate by {n}% after profiling with {s1} tools",
        ],
    },
    "Human Resources": {
        "degree": "business",
        "titles": ["HR Executive", "Talent Acquisition Specialist", "HR Business Partner", "HR Generalist"],
        "skills": ["recruitment", "talent acquisition", "onboarding", "payroll", "employee relations",
                   "HRIS", "Workday", "performance management", "employee engagement", "labour law",
                   "communication", "Excel", "sourcing", "interviewing", "training and development"],
        "duties": [
            "managed end-to-end {s1} for {n} roles across engineering and sales",
            "ran {s1} programmes for new joiners and cut early attrition by {n}%",
            "administered {s1} for {n}0 employees and resolved discrepancies",
            "handled {s1} cases and conducted exit interviews",
            "implemented {s1} to digitise employee records",
            "coordinated quarterly {s1} cycles with department heads",
        ],
    },
    "Sales & Marketing": {
        "degree": "business",
        "titles": ["Digital Marketing Executive", "Sales Executive", "Marketing Manager", "Business Development Associate"],
        "skills": ["digital marketing", "SEO", "Google Ads", "social media marketing", "content marketing",
                   "CRM", "Salesforce", "HubSpot", "lead generation", "negotiation", "market research",
                   "copywriting", "email marketing", "communication", "Excel"],
        "duties": [
            "ran {s1} campaigns that grew qualified leads by {n}%",
            "managed a monthly ad budget across {s1} and {s2}",
            "exceeded quarterly sales targets by {n}% through {s1}",
            "maintained the pipeline in {s1} and forecast monthly revenue",
            "conducted {s1} to identify {n} new customer segments",
            "improved organic traffic by {n}% through {s1} and {s2}",
        ],
    },
    "Finance & Accounting": {
        "degree": "business",
        "titles": ["Financial Analyst", "Accountant", "Audit Associate", "Finance Executive"],
        "skills": ["financial analysis", "financial modeling", "accounting", "GAAP", "IFRS", "Excel",
                   "budgeting", "forecasting", "variance analysis", "taxation", "GST", "auditing",
                   "Tally", "SAP", "Power BI", "bookkeeping", "SQL"],
        "duties": [
            "prepared monthly {s1} reports and presented variances to management",
            "built {s1} models in {s2} to evaluate a {n} crore expansion",
            "managed {s1} and filings, ensuring compliance with deadlines",
            "performed {s1} of {n} client accounts and flagged control gaps",
            "reconciled ledgers in {s1} and closed books within {n} days",
            "automated {s1} reporting with {s2}, saving {n} hours per month",
        ],
    },
}


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _pick(rng: random.Random, items, k):
    return rng.sample(items, k=min(k, len(items)))


# Roles that genuinely overlap in the real world. A resume borrows part of
# its content from one of these neighbours, which is what makes screening hard.
NEIGHBOURS = {
    "Data Science": ["Backend Development", "Finance & Accounting"],
    "Frontend Development": ["Backend Development", "Mobile Development"],
    "Backend Development": ["Frontend Development", "DevOps Engineering", "Data Science"],
    "DevOps Engineering": ["Backend Development"],
    "Mobile Development": ["Frontend Development", "Backend Development"],
    "Human Resources": ["Sales & Marketing"],
    "Sales & Marketing": ["Human Resources", "Finance & Accounting"],
    "Finance & Accounting": ["Data Science", "Sales & Marketing"],
}

GENERIC_TITLES = {
    "tech": ["Software Engineer", "Associate Software Engineer", "Technology Analyst", "Graduate Engineer Trainee"],
    "business": ["Associate", "Executive", "Management Trainee", "Operations Associate"],
}

GENERIC_DUTIES = {
    "tech": [
        "collaborated with product and design teams in two-week sprints",
        "participated in code reviews and wrote technical documentation",
        "fixed production bugs and improved logging and error handling",
        "mentored {n} interns and ran knowledge-sharing sessions",
        "worked with {s1} and {s2} on internal tooling",
    ],
    "business": [
        "prepared weekly reports for senior management",
        "coordinated with cross-functional teams to meet quarterly goals",
        "handled stakeholder communication and documentation",
        "trained {n} new team members on internal processes",
        "used {s1} and {s2} to track team performance",
    ],
}


def make_resume(rng: random.Random, category: str) -> str:
    spec = CATEGORIES[category]
    family = spec["degree"]
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    years = rng.choice([0.5, 1, 1, 2, 2, 3, 3, 4, 5, 6, 8, 10])

    # How much of the resume comes from a neighbouring role (0 = pure, 0.45 = hybrid)
    neighbour = CATEGORIES[rng.choice(NEIGHBOURS[category])]
    blend = rng.choice([0.0, 0.1, 0.2, 0.3, 0.3, 0.4, 0.45])

    core = _pick(rng, spec["skills"], rng.randint(3, 8))
    borrowed = _pick(rng, neighbour["skills"], round(len(core) * blend / (1 - blend)) + rng.randint(0, 2))
    random_other = CATEGORIES[rng.choice([c for c in CATEGORIES if c != category])]
    noise = _pick(rng, random_other["skills"], rng.randint(0, 2))
    skills = list(dict.fromkeys(core + borrowed + noise))
    rng.shuffle(skills)

    def duty() -> str:
        roll = rng.random()
        if roll < blend:
            source, pool = neighbour["duties"], borrowed or core
        elif roll < blend + 0.25:
            source, pool = GENERIC_DUTIES[family], skills
        else:
            source, pool = spec["duties"], core
        s1, s2 = rng.sample(pool, 2) if len(pool) > 1 else (pool[0], pool[0])
        return rng.choice(source).format(s1=s1, s2=s2, n=rng.randint(10, 60))

    def title() -> str:
        roll = rng.random()
        if roll < 0.3:
            return rng.choice(GENERIC_TITLES[family])
        if roll < 0.3 + blend / 2:
            return rng.choice(neighbour["titles"])
        return rng.choice(spec["titles"])

    jobs = []
    for _ in range(rng.randint(1, 3)):
        company = rng.choice(COMPANIES)
        bullets = "\n".join(f"- {_sentence(duty())}" for _ in range(rng.randint(2, 4)))
        jobs.append(f"{title()}, {company}\n{bullets}")

    project = ""
    if borrowed and rng.random() < 0.4:
        project = f"\nPROJECTS\n- Personal project using {', '.join(borrowed[:2])}\n"

    headline = title()
    exp_phrase = f"{years:g} years of experience" if years >= 1 else "6 months of internship experience"
    summary_skills = ", ".join(rng.sample(skills, min(3, len(skills))))
    return (
        f"{name}\n{headline}\n\n"
        f"SUMMARY\n{headline} with {exp_phrase} in {summary_skills}.\n\n"
        f"SKILLS\n{', '.join(skills)}\n\n"
        f"EXPERIENCE\n" + "\n\n".join(jobs) + "\n" + project + "\n"
        f"EDUCATION\n{rng.choice(DEGREES[family])}\n"
    )


def make_job_description(rng: random.Random, category: str) -> str:
    spec = CATEGORIES[category]
    title = rng.choice(spec["titles"])
    required = _pick(rng, spec["skills"], 6)
    nice = _pick(rng, [s for s in spec["skills"] if s not in required], 3)
    years = rng.choice([1, 2, 3, 4, 5])
    duties = [
        d.format(s1=rng.choice(required), s2=rng.choice(required), n=rng.randint(10, 60))
        for d in _pick(rng, spec["duties"], 3)
    ]
    return (
        f"Job Title: {title}\n\n"
        f"We are looking for a {title} with {years}+ years of experience.\n\n"
        f"Responsibilities:\n" + "\n".join(f"- {_sentence(d)}" for d in duties) + "\n\n"
        f"Required skills: {', '.join(required)}\n"
        f"Nice to have: {', '.join(nice)}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-category", type=int, default=60, help="resumes per category (default 60)")
    parser.add_argument("--jds-per-category", type=int, default=5, help="job descriptions per category (default 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    resume_path = DATA_DIR / "resumes.csv"
    with resume_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Resume"])
        for category in CATEGORIES:
            for _ in range(args.per_category):
                writer.writerow([category, make_resume(rng, category)])

    jd_path = DATA_DIR / "job_descriptions.csv"
    with jd_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "JobDescription"])
        for category in CATEGORIES:
            for _ in range(args.jds_per_category):
                writer.writerow([category, make_job_description(rng, category)])

    print(f"Wrote {args.per_category * len(CATEGORIES)} resumes -> {resume_path}")
    print(f"Wrote {args.jds_per_category * len(CATEGORIES)} job descriptions -> {jd_path}")


if __name__ == "__main__":
    main()
