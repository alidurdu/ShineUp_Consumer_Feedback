"""Streamlit app for English-language consumer feedback analysis."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.train_feedback_model import LABEL_ORDER, REVIEW_LABELS, REVIEW_THRESHOLD, build_model, predict_text


APP_DIR = Path(__file__).resolve().parent
LOCAL_DATA = APP_DIR / "data" / "Shineup_Trendyol_250_Reviews_Balanced_EN.xlsx"
LABEL_TITLES = {
    "normal": "Normal",
    "product_issue": "Product issue",
    "possible_sensitivity": "Possible sensitivity",
    "human_review": "Human review",
}
LABEL_COLORS = {
    "Normal": "#4b8f70",
    "Product issue": "#e3a43b",
    "Possible sensitivity": "#d76b63",
    "Human review": "#8064a2",
}

st.set_page_config(
    page_title="Shineup | Consumer Feedback Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_sheets(file_bytes):
    """Load the original and balanced sheets from the English workbook."""
    workbook = load_workbook(BytesIO(file_bytes), read_only=True, data_only=True)
    try:
        if "Original Data" not in workbook.sheetnames:
            raise ValueError("The workbook must include an 'Original Data' sheet.")
        if "Balanced Data" not in workbook.sheetnames:
            raise ValueError("The workbook must include a 'Balanced Data' sheet for machine learning.")

        result = {}
        for name in ("Original Data", "Balanced Data"):
            sheet = workbook[name]
            values = list(sheet.iter_rows(values_only=True))
            if not values:
                raise ValueError(f"The '{name}' sheet is empty.")
            headers = [str(value or "").strip() for value in values[0]]
            result[name] = pd.DataFrame(values[1:], columns=headers).dropna(how="all")
        return result["Original Data"], result["Balanced Data"]
    finally:
        workbook.close()


@st.cache_resource(show_spinner="Training the classifier…")
def train_and_score(texts, labels):
    """Evaluate on an exploratory holdout and return a model trained on all rows."""
    x_train, x_test, y_train, y_test = train_test_split(
        list(texts), list(labels), test_size=0.25, random_state=42, stratify=list(labels)
    )
    evaluation_model = build_model()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    report = classification_report(
        y_test, predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)
    final_model = build_model()
    final_model.fit(list(texts), list(labels))
    metrics = {
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, labels=LABEL_ORDER, average="macro", zero_division=0),
        "report": report,
        "matrix": matrix,
    }
    return final_model, metrics


def show_pie(frame, name_col, value_col, title):
    chart = px.pie(
        frame,
        names=name_col,
        values=value_col,
        hole=0.55,
        color=name_col,
        color_discrete_map=LABEL_COLORS,
        title=title,
    )
    chart.update_traces(textposition="inside", textinfo="percent+label", sort=False)
    chart.update_layout(margin=dict(t=52, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(chart, width="stretch")


st.title("Shineup consumer feedback")
st.caption("A prototype to summarize product reviews, visualize patterns, and prioritize records for review.")

if LOCAL_DATA.is_file():
    default_bytes = LOCAL_DATA.read_bytes()
    st.sidebar.success("English workbook found in the project package.")
else:
    default_bytes = None

uploaded = st.sidebar.file_uploader("Upload the balanced English workbook", type=["xlsx"])
st.sidebar.caption(
    "In the cloud version, an uploaded workbook is sent to the Streamlit server for analysis and classification. "
    "Upload only data you are authorized to share."
)
if uploaded is not None:
    file_bytes = uploaded.getvalue()
elif default_bytes is not None:
    file_bytes = default_bytes
else:
    file_bytes = None

if file_bytes is None:
    st.info("Upload `Shineup_Trendyol_250_Reviews_Balanced_EN.xlsx` to begin.")
    st.markdown(
        "The workbook must include `Original Data` and `Balanced Data` sheets, with the review text in "
        "`Anonymized review summary` and class labels in `Label`. The workbook is kept out of GitHub."
    )
    st.stop()

try:
    original, balanced = load_sheets(file_bytes)
except Exception as error:
    st.error(f"Could not read the workbook: {error}")
    st.stop()

text_col = "Anonymized review summary"
label_col = "Label"
rating_col = "Rating (1–5)" if "Rating (1–5)" in original.columns else "Rating (1-5)"
product_col = "Product / listing"
if text_col not in original.columns or label_col not in original.columns:
    st.error("The Original Data sheet must include 'Anonymized review summary' and 'Label' columns.")
    st.stop()
if text_col not in balanced.columns or label_col not in balanced.columns:
    st.error("The Balanced Data sheet must include 'Anonymized review summary' and 'Label' columns.")
    st.stop()

original = original.dropna(subset=[text_col, label_col]).copy()
balanced = balanced.dropna(subset=[text_col, label_col]).copy()
balanced[text_col] = balanced[text_col].astype(str).str.strip()
balanced[label_col] = balanced[label_col].astype(str).str.strip()
original[label_col] = original[label_col].astype(str).str.strip()
balanced = balanced[balanced[label_col].isin(LABEL_ORDER)]
if balanced.empty or set(balanced[label_col]) != set(LABEL_ORDER):
    st.error("The Balanced Data sheet must contain all four classes.")
    st.stop()

counts = balanced[label_col].value_counts().reindex(LABEL_ORDER, fill_value=0)
model, metrics = train_and_score(tuple(balanced[text_col]), tuple(balanced[label_col]))
original_counts = original[label_col].value_counts().reindex(LABEL_ORDER, fill_value=0)
review_count = int(original[label_col].isin(REVIEW_LABELS).sum())
rating_values = pd.to_numeric(original[rating_col], errors="coerce") if rating_col in original.columns else pd.Series(dtype=float)

st.warning(
    "Original records are shown in the overview. The classifier uses the Balanced Data sheet. "
    "Rephrased examples in that sheet are not independent customer observations, so the metrics are exploratory."
)

overview_tab, model_tab, review_tab, data_tab = st.tabs(
    ["Overview", "Machine learning", "Human review", "Data and method"]
)

with overview_tab:
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Original records", len(original))
    metric2.metric("Products / listings", original[product_col].nunique() if product_col in original.columns else "—")
    metric3.metric("Review queue", review_count)
    metric4.metric("Average rating", f"{rating_values.mean():.2f} / 5" if rating_values.notna().any() else "—")

    st.subheader("Class distribution across the 250 original records")
    left, right = st.columns(2)
    with left:
        label_frame = original_counts.rename_axis("Class").reset_index(name="Records")
        label_frame["Class"] = label_frame["Class"].map(LABEL_TITLES)
        show_pie(label_frame, "Class", "Records", "Review classes")
    with right:
        if rating_col in original.columns:
            rating_frame = rating_values.dropna().value_counts().sort_index().rename_axis("Rating").reset_index(name="Records")
            rating_frame["Rating"] = rating_frame["Rating"].astype(int).astype(str) + " stars"
            show_pie(rating_frame, "Rating", "Records", "Rating distribution")

    if product_col in original.columns:
        products = original[product_col].fillna("Not specified").value_counts().head(8)
        st.subheader("Products with the most reviews")
        st.bar_chart(products)

with model_tab:
    st.subheader("Four-class text classifier")
    c1, c2, c3 = st.columns(3)
    c1.metric("Balanced training rows", len(balanced))
    c2.metric("Training split", metrics["train_rows"])
    c3.metric("Test split", metrics["test_rows"])
    c4, c5 = st.columns(2)
    c4.metric("Holdout accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    c5.metric("Macro F1", f"{metrics['macro_f1']:.3f}")

    class_table = pd.DataFrame({"Class": [LABEL_TITLES[label] for label in LABEL_ORDER], "Examples": counts.values})
    st.dataframe(class_table, hide_index=True, width="stretch")
    matrix_frame = pd.DataFrame(
        metrics["matrix"],
        index=[LABEL_TITLES[label] for label in LABEL_ORDER],
        columns=[LABEL_TITLES[label] for label in LABEL_ORDER],
    )
    st.caption("Confusion matrix · rows: actual class, columns: predicted class")
    st.dataframe(matrix_frame, width="stretch")
    st.caption(
        "No validation set is used. The holdout test split contains 43 rows. Because the balanced data includes "
        "rephrased examples, these scores do not measure performance on independent real-world data."
    )

    st.markdown("#### Try a new review")
    with st.form("predict_form"):
        comment = st.text_area(
            "Anonymized review summary",
            max_chars=2000,
            placeholder="For example: I noticed brief burning on first use.",
        )
        submitted = st.form_submit_button("Classify review", type="primary")
    if submitted:
        if not comment.strip():
            st.info("Enter a review to classify it.")
        else:
            result = predict_text(model, comment.strip())
            probability = result["model_confidence_uncalibrated"]
            if result["review_recommended"]:
                st.warning(
                    f"Prediction: **{LABEL_TITLES[result['predicted_label']]}** · "
                    f"Model confidence: **{probability * 100:.1f}%** · Human review is recommended."
                )
            else:
                st.success(
                    f"Prediction: **{LABEL_TITLES[result['predicted_label']]}** · "
                    f"Model confidence: **{probability * 100:.1f}%**"
                )
            st.caption(result["message"] + " Model confidence is not calibrated.")

with review_tab:
    st.subheader("Original records assigned for human review")
    queue = original[original[label_col].isin(REVIEW_LABELS)].copy()
    if queue.empty:
        st.info("There are no records in the review queue.")
    else:
        filter_label = st.selectbox("Filter by class", ["All", "Possible sensitivity", "Human review"])
        if filter_label != "All":
            selected = {title: label for label, title in LABEL_TITLES.items()}[filter_label]
            queue = queue[queue[label_col] == selected]
        search_text = st.text_input("Search review text")
        if search_text:
            queue = queue[queue[text_col].str.contains(search_text, case=False, na=False)]
        display = queue.copy()
        display[label_col] = display[label_col].map(LABEL_TITLES)
        st.dataframe(display, hide_index=True, width="stretch")
    st.info("Labels do not represent confirmed health events or diagnoses. A qualified reviewer should assess the records.")

with data_tab:
    st.subheader("Source and method")
    st.write(f"Original data: **{len(original)} records**")
    st.write(f"Balanced data: **{len(balanced)} records** · 43 examples per class")
    st.write("Model: TF-IDF text features and logistic regression.")
    st.write("Evaluation: stratified 25% holdout (43 rows). No separate validation split is used.")
    st.write("The app reads the workbook into memory for analysis and classification. The code does not write it to a database or save it to disk.")
    st.dataframe(original.head(20), hide_index=True, width="stretch")
    st.caption("Original review summaries and class labels are translated from the source workbook.")

st.divider()
st.caption("Training and decision-support prototype · Not an automated health diagnosis or treatment recommendation")
