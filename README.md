# MLOps Assignment 01 — Heart Disease Classification

A reproducible starter implementation for the MLOps AIMLCZG523 assignment. **Customize the experiments, screenshots, report, and deployment for your own submission**; the assignment explicitly requires original work and understanding.

## Stack
- Python, pandas, scikit-learn
- MLflow for experiment tracking
- FastAPI for model serving
- Pytest + Ruff for testing/linting
- Docker for packaging
- GitHub Actions for CI
- Kubernetes manifests for deployment
- Prometheus-compatible `/metrics` endpoint

## Project structure
```text
mlops-heart-disease/
├── data/                       # downloaded dataset (generated)
├── artifacts/                  # trained model (generated)
├── reports/                    # plots and report material (generated)
├── src/
│   ├── data.py
│   ├── train.py
│   └── api.py
├── tests/
│   ├── test_data.py
│   └── test_api.py
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 1. Setup (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download and inspect data
```bash
python -m src.data
```
This downloads the processed Cleveland heart disease dataset from the UCI repository, saves a CSV, and generates basic EDA plots. Internet access is required for the first download.

The original dataset uses `?` for some missing values. The code converts these to missing values and the training pipeline imputes them. The target is converted to binary: `0` = no disease, `1–4` = presence of disease. This is a modelling exercise, not a clinical diagnostic tool.

## 3. Train models and log MLflow runs
```bash
mlflow ui --host 127.0.0.1 --port 5000
# In another terminal:
python -m src.train
```
Open http://127.0.0.1:5000 to inspect runs. The training script compares Logistic Regression and Random Forest using stratified cross-validation, logs parameters/metrics/artifacts, and saves the selected full preprocessing-and-model pipeline to `artifacts/model.joblib`.

## 4. Run API locally
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Open http://127.0.0.1:8000/docs for interactive API documentation.

Example request (all 13 input features are required):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
  }'
```
The API returns a binary prediction and the model's estimated class probability. The probability is not a clinically validated risk estimate.

## 5. Test and lint
```bash
pytest -q
ruff check .
```

## 6. Build and run Docker
Train the model first so `artifacts/model.joblib` exists.
```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```
Then test `http://localhost:8000/health`, `/docs`, and `/predict`.

## 7. Kubernetes (local Minikube or Docker Desktop)
Build the image in the Kubernetes runtime or use a registry accessible to the cluster. For Minikube:
```bash
minikube image load heart-disease-api:local
kubectl apply -f k8s/
kubectl get pods
kubectl get svc
kubectl port-forward svc/heart-disease-api 8000:80
```
For Docker Desktop, ensure Kubernetes is enabled and the image is available to the cluster. The included service is `NodePort`; port-forwarding is the simplest local verification method. A public cloud deployment requires a cloud account, image registry, cluster/service configuration, and actual deployment evidence—do not claim it is public unless you deploy it.

## 8. Evidence to collect yourself
1. EDA plots from `reports/`.
2. MLflow UI showing at least two runs and logged metrics.
3. `pytest` and `ruff` passing.
4. GitHub Actions successful workflow run.
5. Docker build and running container, plus `/health` and `/predict` responses.
6. Kubernetes `kubectl get pods,svc` and API response.
7. Monitoring evidence from `/metrics` or Prometheus/Grafana if configured.
8. A short screen recording demonstrating the pipeline.
9. A ten-page report based on your actual results, architecture, screenshots, limitations, and repository URL.

## Important limitations
- Dataset download requires internet access.
- Training metrics are not hard-coded; report the actual values from your run.
- The included Kubernetes manifests are a local deployment starting point, not proof of cloud deployment.
- Do not use this model for medical decisions. It is an educational assignment.
