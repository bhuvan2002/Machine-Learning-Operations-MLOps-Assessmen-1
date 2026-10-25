import json
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import create_eda, load_data

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
TARGET = "target"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def build_pipeline(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    create_eda(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_experiment("heart-disease-classification")

    candidates = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
        ),
    }

    results = []
    for name, pipeline in candidates.items():
        with mlflow.start_run(run_name=name):
            scores = cross_validate(
                pipeline, X_train, y_train, cv=cv,
                scoring={"accuracy": "accuracy", "precision": "precision",
                         "recall": "recall", "roc_auc": "roc_auc"},
                return_train_score=False,
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = {
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_precision_mean": float(scores["test_precision"].mean()),
                "cv_recall_mean": float(scores["test_recall"].mean()),
                "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
                "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            }
            mlflow.log_param("model", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics(metrics)
            mlflow.log_text(classification_report(y_test, predictions), "classification_report.txt")

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax)
            fig.tight_layout()
            cm_path = REPORTS / f"{name}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
            fig.tight_layout()
            roc_path = REPORTS / f"{name}_roc_curve.png"
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(
    pipeline,
    name="model",
    serialization_format="cloudpickle",
)
            results.append((name, metrics, pipeline))

    # Choose model by mean cross-validation ROC-AUC, not by test-set performance.
    best_name, best_metrics, best_pipeline = max(
        results, key=lambda item: item[1]["cv_roc_auc_mean"]
    )
    joblib.dump(best_pipeline, ARTIFACTS / "model.joblib")
    summary = {
        "selected_model": best_name,
        "selection_rule": "highest mean 5-fold cross-validation ROC-AUC",
        "models": {name: metrics for name, metrics, _ in results},
    }
    (REPORTS / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved selected pipeline to {ARTIFACTS / 'model.joblib'}")


if __name__ == "__main__":
    main()
