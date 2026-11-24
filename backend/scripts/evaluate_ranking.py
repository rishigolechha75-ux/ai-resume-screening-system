"""Evaluate how well the ranker puts relevant resumes at the top.

Setup: the classifier is trained on 80% of the resumes. For each job
description, the remaining 20% (unseen) resumes are ranked. A resume counts
as relevant when its category matches the job description's category.

Three rankers are compared:
  * TF-IDF only   - cosine similarity alone (baseline)
  * Skills only   - skill coverage alone
  * Hybrid        - the full weighted score used by the API

Metrics: Precision@5, Precision@10 and NDCG@10, averaged over all job descriptions.

Usage:
    python scripts/evaluate_ranking.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier import DATA_PATH, CategoryClassifier, load_dataset  # noqa: E402
from app.ranker import ResumeInput, rank_resumes  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent
JD_PATH = BACKEND_DIR / "data" / "job_descriptions.csv"
REPORTS_DIR = BACKEND_DIR / "reports"

RANKERS = {
    "TF-IDF only": {"similarity": 1.0, "skills": 0.0, "category": 0.0, "experience": 0.0},
    "Skills only": {"similarity": 0.0, "skills": 1.0, "category": 0.0, "experience": 0.0},
    "Hybrid (default)": None,
}


def precision_at(relevance, k):
    return sum(relevance[:k]) / k


def ndcg_at(relevance, k):
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    texts, labels = load_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    classifier = CategoryClassifier().fit(x_train, y_train)
    pool = [ResumeInput(name=f"r{i}", text=t) for i, t in enumerate(x_test)]
    category_of = {f"r{i}": c for i, c in enumerate(y_test)}

    jds = pd.read_csv(JD_PATH)
    scores = {name: {"P@5": [], "P@10": [], "NDCG@10": []} for name in RANKERS}

    for _, row in jds.iterrows():
        for name, weights in RANKERS.items():
            ranked = rank_resumes(row["JobDescription"], pool, classifier, weights)["results"]
            relevance = [1 if category_of[r["name"]] == row["Category"] else 0 for r in ranked]
            scores[name]["P@5"].append(precision_at(relevance, 5))
            scores[name]["P@10"].append(precision_at(relevance, 10))
            scores[name]["NDCG@10"].append(ndcg_at(relevance, 10))

    summary = {
        name: {metric: round(sum(v) / len(v), 4) for metric, v in metrics.items()}
        for name, metrics in scores.items()
    }
    print(f"Ranked {len(pool)} unseen resumes for each of {len(jds)} job descriptions\n")
    print(f"{'Ranker':<18}{'P@5':>8}{'P@10':>8}{'NDCG@10':>10}")
    for name, m in summary.items():
        print(f"{name:<18}{m['P@5']:>8.3f}{m['P@10']:>8.3f}{m['NDCG@10']:>10.3f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "ranking_metrics.json").write_text(
        json.dumps(
            {"n_job_descriptions": len(jds), "n_resumes_ranked": len(pool), "results": summary},
            indent=2,
        )
    )
    print(f"\nSaved metrics -> {REPORTS_DIR / 'ranking_metrics.json'}")


if __name__ == "__main__":
    main()
