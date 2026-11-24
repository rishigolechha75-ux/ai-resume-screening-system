"""Job-category classifier: TF-IDF features + Logistic Regression.

The classifier predicts which job category a resume (or a job description)
belongs to. The ranker uses it to check whether a candidate's profile fits
the role family of the job being screened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import clean_text

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BACKEND_DIR / "data" / "resumes.csv"
MODEL_PATH = BACKEND_DIR / "models" / "category_classifier.joblib"
METADATA_PATH = BACKEND_DIR / "models" / "category_classifier.json"


def build_pipeline() -> Pipeline:
    """Create an untrained TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=5.0)),
        ]
    )


def load_dataset(path: Path = DATA_PATH) -> Tuple[List[str], List[str]]:
    """Load a CSV with ``Category`` and ``Resume`` columns."""
    df = pd.read_csv(path)
    missing = {"Category", "Resume"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset {path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["Category", "Resume"])
    return df["Resume"].astype(str).tolist(), df["Category"].astype(str).tolist()


class CategoryClassifier:
    """Thin wrapper around the scikit-learn pipeline with save/load helpers."""

    def __init__(self, pipeline: Optional[Pipeline] = None):
        self.pipeline = pipeline

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None and hasattr(self.pipeline.named_steps["clf"], "classes_")

    @property
    def categories(self) -> List[str]:
        return list(self.pipeline.named_steps["clf"].classes_) if self.is_trained else []

    def fit(self, texts: List[str], labels: List[str]) -> "CategoryClassifier":
        self.pipeline = build_pipeline()
        self.pipeline.fit(texts, labels)
        return self

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return ``{category: probability}`` sorted from most to least likely."""
        probs = self.pipeline.predict_proba([text])[0]
        pairs = sorted(zip(self.categories, probs), key=lambda p: p[1], reverse=True)
        return {str(c): float(p) for c, p in pairs}

    def predict(self, text: str) -> Tuple[str, float]:
        """Return the most likely category and its probability."""
        category, prob = next(iter(self.predict_proba(text).items()))
        return category, prob

    def save(self, path: Path = MODEL_PATH, metadata: Optional[dict] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        if metadata is not None:
            METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CategoryClassifier":
        return cls(joblib.load(path))


def load_or_train(model_path: Path = MODEL_PATH, data_path: Path = DATA_PATH) -> CategoryClassifier:
    """Load the saved model, or train one from the bundled dataset if none exists."""
    if model_path.exists():
        return CategoryClassifier.load(model_path)
    texts, labels = load_dataset(data_path)
    model = CategoryClassifier().fit(texts, labels)
    model.save(model_path)
    return model
