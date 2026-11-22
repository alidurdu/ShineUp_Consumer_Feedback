"""Training-oriented text classifier for consumer feedback.

The model trains on the balanced workbook sheet. Possible sensitivity and
uncertain records are routed for human review. The model does not diagnose or
establish causality.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
from openpyxl import load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABEL_ORDER = ["normal", "product_issue", "possible_sensitivity", "human_review"]
REVIEW_LABELS = {"possible_sensitivity", "human_review"}
REVIEW_THRESHOLD = 0.55


def normalize_header(value):
    """Normalize a spreadsheet header for reliable matching."""
    text = unicodedata.normalize("NFKD", str(value or "").strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_training_data(path):
    """Read review text and labels from the workbook's Balanced Data sheet."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if "Balanced Data" not in workbook.sheetnames:
            raise ValueError("The workbook does not contain a 'Balanced Data' sheet.")
        sheet = workbook["Balanced Data"]
        rows = sheet.iter_rows(values_only=True)
        raw_headers = next(rows, None)
        if not raw_headers:
            raise ValueError("The 'Balanced Data' sheet is empty.")
        headers = [normalize_header(value) for value in raw_headers]
        text_col = next((i for i, name in enumerate(headers) if name == "anonymized review summary"), None)
        label_col = next((i for i, name in enumerate(headers) if name == "label"), None)
        if text_col is None or label_col is None:
            raise ValueError("The review text or Label column is missing.")

        texts, labels = [], []
        for row_number, row in enumerate(rows, start=2):
            text = str(row[text_col] or "").strip()
            label = str(row[label_col] or "").strip()
            if not text and not label:
                continue
            if not text or label not in LABEL_ORDER:
                raise ValueError(f"Row {row_number} has empty text or an invalid class label: {label!r}")
            texts.append(text)
            labels.append(label)
        if len(set(labels)) != len(LABEL_ORDER):
            raise ValueError("All four expected classes must be present in the training data.")
        return texts, labels
    finally:
        workbook.close()


def build_model():
    """Build a TF-IDF text pipeline followed by logistic regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def evaluate_demo_model(texts, labels, output_dir):
    """Write exploratory holdout metrics for the balanced training sheet."""
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER).tolist()
    result = {
        "evaluation_type": "exploratory_holdout_on_balanced_training_sheet",
        "test_rows": len(y_test),
        "labels_in_order": LABEL_ORDER,
        "classification_report": report,
        "confusion_matrix": matrix,
        "limitation": (
            "The balanced sheet includes rephrased examples derived from original records. "
            "This random split is not an independent real-world evaluation, and its metrics "
            "do not establish performance on actual customer feedback."
        ),
    }
    (output_dir / "demo_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def predict_text(model, text):
    """Classify a review and indicate whether human review is recommended."""
    probabilities = model.predict_proba([text])[0]
    best_index = probabilities.argmax()
    label = str(model.named_steps["classifier"].classes_[best_index])
    confidence = float(probabilities[best_index])
    return {
        "text": text,
        "predicted_label": label,
        "model_confidence_uncalibrated": round(confidence, 4),
        "review_recommended": label in REVIEW_LABELS or confidence < REVIEW_THRESHOLD,
        "message": "This classification supports prioritization only; it is not a health diagnosis.",
    }


def main():
    parser = argparse.ArgumentParser(description="Train a text classifier using balanced feedback data.")
    parser.add_argument("--input", required=True, help="Excel workbook containing a Balanced Data sheet")
    parser.add_argument("--output", default="outputs/feedback_model", help="Directory for the model and metrics")
    parser.add_argument("--text", help="Optional new review to classify after training")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_path.is_file():
        parser.error(f"Input workbook not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = load_training_data(input_path)
    counts = Counter(labels)
    metrics = evaluate_demo_model(texts, labels, output_dir)

    final_model = build_model()
    final_model.fit(texts, labels)
    model_path = output_dir / "feedback_text_model.joblib"
    joblib.dump(final_model, model_path)

    print("Training complete.")
    print(f"Records: {len(texts)} | Classes: {dict(counts)}")
    print(f"Exploratory metrics: {output_dir / 'demo_metrics.json'}")
    print(f"Trained model: {model_path}")
    print(f"Evaluation note: {metrics['limitation']}")
    if args.text:
        print(json.dumps(predict_text(final_model, args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
