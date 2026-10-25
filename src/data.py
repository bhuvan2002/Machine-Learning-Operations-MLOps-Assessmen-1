from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DATA_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_data(download: bool = True) -> pd.DataFrame:
    """Load the processed Cleveland dataset; download the UCI ZIP if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "heart_disease.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif download:
        with urlopen(DATA_URL, timeout=30) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive:
            member = next(
                (n for n in archive.namelist() if n.lower().endswith("processed.cleveland.data")),
                None,
            )
            if member is None:
                raise FileNotFoundError("processed.cleveland.data was not found in UCI archive")
            with archive.open(member) as raw:
                df = pd.read_csv(raw, header=None, names=COLUMNS, na_values="?")
        df.to_csv(csv_path, index=False)
    else:
        raise FileNotFoundError(f"Dataset not found at {csv_path}; run with download=True")

    # Ensure the target is binary: 0 means no disease; 1–4 mean disease present.
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = (df["target"] > 0).astype(int)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def create_eda(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.hist(figsize=(14, 10), bins=20)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_histograms.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="target")
    plt.title("Class distribution (0 = no disease, 1 = disease)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_balance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    dataset = load_data()
    create_eda(dataset)
    print(f"Rows: {len(dataset)}; columns: {len(dataset.columns)}")
    print("Missing values by column:\n", dataset.isna().sum())
    print("EDA plots saved in reports/")
