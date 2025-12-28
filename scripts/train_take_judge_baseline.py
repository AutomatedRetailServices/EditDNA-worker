"""
Train a SIMPLE baseline take_judge model from the dataset we built.

It:
- Downloads the JSONL dataset from S3:
      s3://{EDITDNA_DATASET_BUCKET}/{EDITDNA_TAKEJUDGE_DATASET_KEY}
  default: script2clipshop-video-automatedretailservices / editdna/training/take_judge_dataset.jsonl

- Uses "text" as input and "label" (0/1) as output
- Trains a TF-IDF + Logistic Regression classifier
- Saves the trained model (vectorizer + classifier) to S3 as:
      s3://{EDITDNA_DATASET_BUCKET}/{EDITDNA_TAKEJUDGE_MODEL_KEY}
  default: editdna/models/take_judge_baseline.joblib
"""

import os
import json
from typing import List, Dict, Any, Tuple

import boto3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


# ================== CONFIG ==================

BUCKET_NAME = os.getenv(
    "EDITDNA_DATASET_BUCKET",
    "script2clipshop-video-automatedretailservices",
)

DATASET_KEY = os.getenv(
    "EDITDNA_TAKEJUDGE_DATASET_KEY",
    "editdna/training/take_judge_dataset.jsonl",
)

MODEL_KEY = os.getenv(
    "EDITDNA_TAKEJUDGE_MODEL_KEY",
    "editdna/models/take_judge_baseline.joblib",
)

LOCAL_DATASET_PATH = "/tmp/take_judge_dataset.jsonl"
LOCAL_MODEL_PATH = "/tmp/take_judge_baseline.joblib"


# ================== HELPERS ==================


def download_dataset_from_s3() -> None:
    s3 = boto3.client("s3")
    print(f"Downloading dataset from s3://{BUCKET_NAME}/{DATASET_KEY}")
    os.makedirs(os.path.dirname(LOCAL_DATASET_PATH), exist_ok=True)
    s3.download_file(BUCKET_NAME, DATASET_KEY, LOCAL_DATASET_PATH)
    print(f"Saved local dataset to {LOCAL_DATASET_PATH}")


def load_dataset() -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    print(f"Loading dataset from {LOCAL_DATASET_PATH}")
    with open(LOCAL_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = (row.get("text") or "").strip()
            if not text:
                continue

            # 0 or 1
            label = int(row.get("label", 0))

            texts.append(text)
            labels.append(label)

    print(f"Loaded {len(texts)} clips with labels")
    return texts, labels


def train_model(texts: List[str], labels: List[int]):
    print("Splitting train / validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )

    print(f"Train size: {len(X_train)}  |  Val size: {len(X_val)}")

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train_vec, y_train)

    print("Validation performance:")
    y_pred = clf.predict(X_val_vec)
    print(classification_report(y_val, y_pred, digits=3))

    return vectorizer, clf


def save_and_upload_model(vectorizer, clf) -> None:
    print(f"Saving model to {LOCAL_MODEL_PATH}")
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

    bundle = {
        "vectorizer": vectorizer,
        "classifier": clf,
    }

    joblib.dump(bundle, LOCAL_MODEL_PATH)

    print(f"Uploading model to s3://{BUCKET_NAME}/{MODEL_KEY}")
    s3 = boto3.client("s3")
    with open(LOCAL_MODEL_PATH, "rb") as f:
        s3.put_object(Bucket=BUCKET_NAME, Key=MODEL_KEY, Body=f.read())

    print("Model upload DONE ✅")


# ================== MAIN ==================


def main():
    print("=== TAKE_JUDGE TRAINING START ===")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Dataset key: {DATASET_KEY}")
    print(f"Model key: {MODEL_KEY}")

    download_dataset_from_s3()
    texts, labels = load_dataset()

    if not texts:
        print("ERROR: No data loaded. Check dataset file.")
        return

    vectorizer, clf = train_model(texts, labels)
    save_and_upload_model(vectorizer, clf)

    print("=== TAKE_JUDGE TRAINING COMPLETE ✅ ===")


if __name__ == "__main__":
    main()
