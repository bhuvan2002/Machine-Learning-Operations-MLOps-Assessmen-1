import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.train import build_pipeline


def test_pipeline_fits_with_missing_values():
    from src.train import CATEGORICAL, NUMERIC
    columns = NUMERIC + CATEGORICAL
    row = {
        "age": 55, "trestbps": 130, "chol": 250, "thalach": 150, "oldpeak": 1.0,
        "sex": 1, "cp": 2, "fbs": 0, "restecg": 1, "exang": 0,
        "slope": 2, "ca": 0, "thal": 3,
    }
    frame = pd.DataFrame([row, {**row, "chol": None, "thal": None}], columns=columns)
    labels = [0, 1]
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(frame, labels)
    assert len(pipe.predict(frame)) == 2
