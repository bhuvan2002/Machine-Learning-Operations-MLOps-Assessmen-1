# MLOps Assignment 01: Heart Disease Classification

> Replace every bracketed placeholder with results from your own execution. Add your own screenshots and explanations. Do not report metrics or deployment status that you have not verified.

## 1. Abstract
[Summarize the problem, approach, best model, measured result, deployment method, and limitations.]

## 2. Problem Statement and Objectives
[Describe binary classification and the MLOps goals.]

## 3. Dataset Acquisition and Description
- Source: UCI Machine Learning Repository, Heart Disease dataset (processed Cleveland subset).
- Rows / features after loading: [actual values]
- Target conversion: 0 remains 0; original values 1–4 become 1.
- Missing values: [actual counts and handling]
- Dataset source and download method: [details]

## 4. Exploratory Data Analysis
Insert `feature_histograms.png`, `correlation_heatmap.png`, and `class_balance.png`.
[Describe only patterns visible in your generated plots.]

## 5. Preprocessing and Feature Engineering
[Explain numerical median imputation + scaling and categorical most-frequent imputation + one-hot encoding. Discuss why the preprocessing is fitted inside the pipeline to reduce leakage.]

## 6. Models and Experimental Method
[Describe Logistic Regression and Random Forest, stratified 5-fold cross-validation, random seed, train/test split, and any additional tuning you actually ran.]

## 7. Results and Model Selection
Insert actual values from `reports/metrics_summary.json`.
| Model | CV accuracy | CV precision | CV recall | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

[Explain the selection rule and trade-offs. Do not choose based only on accuracy.]

## 8. MLflow Experiment Tracking
[Add screenshot of MLflow runs. Explain logged parameters, metrics, plots, and model artifacts.]

## 9. Packaging and Reproducibility
[Explain requirements.txt, fixed random seeds, saved complete scikit-learn pipeline, and clean setup steps.]

## 10. CI/CD, Containerization, Deployment, Monitoring
[Include GitHub Actions success screenshot, pytest output, Docker build/run screenshots, API request/response, Kubernetes evidence, and `/metrics` output. State honestly whether deployment is local or public.]

## 11. Architecture
```text
UCI dataset -> preprocessing + training -> MLflow runs
            -> selected pipeline artifact -> FastAPI -> Docker
GitHub push -> Actions: lint -> tests -> train -> artifacts
FastAPI -> request logs + Prometheus metrics
Docker image -> Kubernetes Deployment -> Service -> client
```

## 12. Limitations, Ethics, and Future Work
[Small/older dataset, possible sampling bias, no clinical validation, probability calibration, model drift, data privacy, monitoring alerts, and larger external validation.]

## 13. Conclusion
[Summarize what you actually implemented and learned.]

## References
- UCI Machine Learning Repository, Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
- scikit-learn documentation: https://scikit-learn.org/
- MLflow documentation: https://mlflow.org/docs/latest/
- FastAPI documentation: https://fastapi.tiangolo.com/
- Docker documentation: https://docs.docker.com/
- Kubernetes documentation: https://kubernetes.io/docs/
