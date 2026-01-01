# bank_mlops_pipeline.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# ---------- Paths ----------
DATA_PATH = "bank-additional.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gb_bank_pipeline.joblib")


# ---------- Data ingestion ----------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    return df


# ---------- Pre‑processing ----------
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.replace("unknown", np.nan)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})
    return X, y


# ---------- Feature engineering / preprocessor ----------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    # Column name can be day_of_week or dayofweek depending on file
    day_col = "day_of_week" if "day_of_week" in X.columns else "dayofweek"
    categorical_features = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        day_col,
        "poutcome",
    ]
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False  # use sparse=False if on older sklearn
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# ---------- Model ----------
def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", gb),
        ]
    )
    return pipe


# ---------- Evaluation helpers ----------
def plot_basic_eda(df: pd.DataFrame):
    # Target distribution
    plt.figure(figsize=(4, 3))
    df["y"].value_counts().plot(kind="bar")
    plt.title("Target distribution (y)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png")
    plt.close()

    # Duration vs target
    plt.figure(figsize=(4, 3))
    sns.boxplot(x="y", y="duration", data=df)
    plt.title("Call duration vs term deposit")
    plt.tight_layout()
    plt.savefig("eda_duration_vs_y.png")
    plt.close()


def evaluate_model(model: Pipeline, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Gradient Boosting")
    plt.tight_layout()
    plt.savefig("confusion_matrix_gb.png")
    plt.close()

    return cm, auc


# ---------- Persistence ----------
def save_model(model: Pipeline):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved full pipeline to {MODEL_PATH}")


def load_model(path: str = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


# ---------- Main ----------
def main():
    # 1. Ingestion
    df = load_data(DATA_PATH)

    # 2. Cleaning / preprocessing
    df = basic_cleaning(df)

    # Optional: quick EDA plots
    plot_basic_eda(df)

    # 3. Features / target
    X, y = split_features_target(df)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Preprocessor + model
    preprocessor = build_preprocessor(X_train)
    model = build_model(preprocessor)

    # 6. Train
    model.fit(X_train, y_train)

    # 7. Evaluate (report, ROC-AUC, confusion matrix image)
    cm, auc = evaluate_model(model, X_test, y_test)
    print("Confusion matrix:\n", cm)
    print("ROC-AUC:", auc)

    # 8. Persist
    save_model(model)


if __name__ == "__main__":
    main()
