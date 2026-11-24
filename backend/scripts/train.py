"""Train and evaluate the job-category classifier.

Compares three classic text classifiers on a stratified 80/20 split plus
5-fold cross-validation, then trains the chosen model (TF-IDF + Logistic
Regression) on the full dataset and saves it to ``models/``.

Usage:
    python scripts/train.py                          # bundled synthetic dataset
    python scripts/train.py --data path/to/file.csv  # any CSV with Category,Resume columns
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classifier import DATA_PATH, MODEL_PATH, CategoryClassifier, build_pipeline, load_dataset  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def candidate_models() -> dict:
    base = build_pipeline()
    tfidf = base.named_steps["tfidf"]
    return {
        "Logistic Regression": base,
        "Linear SVM": Pipeline([("tfidf", clone(tfidf)), ("clf", LinearSVC(C=1.0))]),
        "Multinomial Naive Bayes": Pipeline([("tfidf", clone(tfidf)), ("clf", MultinomialNB(alpha=0.1))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV with Category and Resume columns")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    texts, labels = load_dataset(args.data)
    print(f"Loaded {len(texts)} resumes across {len(set(labels))} categories from {args.data}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=args.seed
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    comparison = {}
    for name, model in candidate_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        cv_scores = cross_val_score(clone(model), texts, labels, cv=folds, scoring="accuracy")
        comparison[name] = {
            "test_accuracy": round(accuracy_score(y_test, predictions), 4),
            "test_macro_f1": round(f1_score(y_test, predictions, average="macro"), 4),
            "cv5_accuracy_mean": round(cv_scores.mean(), 4),
            "cv5_accuracy_std": round(cv_scores.std(), 4),
        }
        print(f"{name:<26} test acc={comparison[name]['test_accuracy']:.3f}  "
              f"macro-F1={comparison[name]['test_macro_f1']:.3f}  "
              f"5-fold acc={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    chosen = candidate_models()["Logistic Regression"].fit(x_train, y_train)
    report = classification_report(y_test, chosen.predict(x_test), output_dict=True)
    print("\nPer-category results (Logistic Regression, held-out 20%):")
    print(classification_report(y_test, chosen.predict(x_test)))

    final = CategoryClassifier().fit(texts, labels)
    metadata = {
        "model": "TF-IDF (1-2 grams) + Logistic Regression",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.data.name),
        "n_samples": len(texts),
        "categories": final.categories,
        "comparison": comparison,
    }
    final.save(MODEL_PATH, metadata)
    print(f"Saved model -> {MODEL_PATH}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "classifier_metrics.json").write_text(
        json.dumps({"comparison": comparison, "per_category": report}, indent=2)
    )
    print(f"Saved metrics -> {REPORTS_DIR / 'classifier_metrics.json'}")


if __name__ == "__main__":
    main()
