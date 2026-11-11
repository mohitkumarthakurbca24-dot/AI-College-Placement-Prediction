import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "placement_data.csv")

FEATURES = [
    "cgpa","tenth_percentage","twelfth_percentage","backlogs",
    "programming_skill","aptitude_score","communication_score",
    "internships","certifications","projects","attendance_percentage",
    "technical_interview_score","mock_interview_score"
]

def make_demo_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cgpa": np.round(rng.uniform(5.0, 10.0, n), 2),
        "tenth_percentage": np.round(rng.uniform(50, 98, n), 1),
        "twelfth_percentage": np.round(rng.uniform(50, 98, n), 1),
        "backlogs": rng.integers(0, 5, n),
        "programming_skill": rng.integers(1, 11, n),
        "aptitude_score": rng.integers(30, 101, n),
        "communication_score": rng.integers(30, 101, n),
        "internships": rng.integers(0, 4, n),
        "certifications": rng.integers(0, 8, n),
        "projects": rng.integers(0, 7, n),
        "attendance_percentage": np.round(rng.uniform(55, 100, n), 1),
        "technical_interview_score": rng.integers(30, 101, n),
        "mock_interview_score": rng.integers(30, 101, n),
    })
    score = (
        1.4*df.cgpa + 0.018*df.tenth_percentage + 0.018*df.twelfth_percentage
        - 0.55*df.backlogs + 0.16*df.programming_skill
        + 0.012*df.aptitude_score + 0.012*df.communication_score
        + 0.35*df.internships + 0.08*df.certifications + 0.14*df.projects
        + 0.012*df.attendance_percentage
        + 0.012*df.technical_interview_score + 0.012*df.mock_interview_score
        + rng.normal(0, 1.5, n)
    )
    threshold = np.quantile(score, 0.55)
    df["placed"] = (score >= threshold).astype(int)
    return df

def train():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        missing = [c for c in FEATURES + ["placed"] if c not in df.columns]
        if missing:
            raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    else:
        df = make_demo_data()
        df.to_csv(DATA_FILE, index=False)

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(df[FEATURES].median(numeric_only=True))
    y = pd.to_numeric(df["placed"], errors="coerce").fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "training_rows": len(X_train),
        "test_rows": len(X_test)
    }

    joblib.dump({"model": model, "features": FEATURES}, os.path.join(MODEL_DIR, "placement_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        import json; json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train()
