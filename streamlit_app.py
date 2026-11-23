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


def rank_products(original, product_col, rating_col, label_col, minimum_reviews=10):
    """Rank products using original review ratings and the existing review labels."""
    if product_col not in original.columns or rating_col not in original.columns:
        return pd.DataFrame()

    reviews = original[[product_col, rating_col, label_col]].copy()
    reviews[product_col] = reviews[product_col].fillna("").astype(str).str.strip()
    reviews["_rating"] = pd.to_numeric(reviews[rating_col], errors="coerce")
    reviews = reviews[(reviews[product_col] != "") & reviews["_rating"].notna()]
    if reviews.empty:
        return pd.DataFrame()

    ranked = reviews.groupby(product_col).agg(
        review_count=("_rating", "size"),
        average_rating=("_rating", "mean"),
        normal_count=(label_col, lambda values: (values == "normal").sum()),
        product_issue_count=(label_col, lambda values: (values == "product_issue").sum()),
        review_priority_count=(label_col, lambda values: values.isin(REVIEW_LABELS).sum()),
    )
    ranked["normal_share"] = ranked["normal_count"] / ranked["review_count"]
    ranked["review_score"] = 100 * (
        0.70 * ranked["average_rating"] / 5 + 0.30 * ranked["normal_share"]
    )
    ranked = ranked[ranked["review_count"] >= minimum_reviews]
    return ranked.sort_values(
        ["review_score", "review_count", "average_rating"], ascending=False
    ).reset_index(names="product")


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
        st.subheader("Top 3 products by review score")
        ranked_products = rank_products(original, product_col, rating_col, label_col)
        if ranked_products.empty:
            st.info("There are not enough rated product reviews to create a ranking.")
        else:
            top_products = ranked_products.head(3).copy()
            top_products = top_products.rename(
                columns={
                    "product": "Product",
                    "review_count": "Reviews",
                    "average_rating": "Average rating / 5",
                    "normal_share": "Normal-label share",
                    "product_issue_count": "Product-issue reviews",
                    "review_priority_count": "Human-review flags",
                    "review_score": "Review score / 100",
                }
            )
            top_products.insert(0, "Rank", range(1, len(top_products) + 1))
            score_chart = px.bar(
                top_products,
                x="Review score / 100",
                y="Product",
                orientation="h",
                text="Review score / 100",
                color="Review score / 100",
                color_continuous_scale=["#dce9df", "#4b8f70"],
                range_x=[0, 100],
            )
            score_chart.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            score_chart.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Composite review score (0–100)",
                yaxis_title="",
                yaxis={"categoryorder": "total ascending"},
                margin=dict(t=12, b=16, l=12, r=60),
                height=260,
            )
            st.plotly_chart(score_chart, width="stretch")
            st.dataframe(
                top_products[
                    ["Rank", "Product", "Reviews", "Average rating / 5", "Normal-label share", "Product-issue reviews", "Human-review flags"]
                ].style.format(
                    {"Average rating / 5": "{:.2f}", "Normal-label share": "{:.1%}"}
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Score = 70% average rating (normalized to 5) + 30% share labeled Normal. "
                "Only products with at least 10 rated original reviews are ranked. This is an exploratory "
                "sample-based shortlist, not a product-quality or safety determination. "
                "Human-review flags include possible-sensitivity and uncertain records."
            )

        products = original[product_col].fillna("Not specified").value_counts().head(8)
        st.subheader("Most reviewed products")
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
