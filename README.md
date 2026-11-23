# Shineup Consumer Feedback Analytics

This project includes an English-language Streamlit app, an English translation of the review workbook, and a four-class text-classification prototype. The app summarizes the original review records, shows class and rating charts, highlights records for human review, and lets a user try a text prediction.

## Live app

Open the deployed application: [Shineup Consumer Health Analytics](https://shineup-consumer-health-analytics.streamlit.app/).

## Data workbook

The translated workbook is `data/Shineup_Trendyol_250_Reviews_Balanced_EN.xlsx`.

- `Original Data` contains the 250 source records with English column labels, translated product descriptions, and translated review summaries. Source URLs and dates are retained.
- `Balanced Data` contains 172 rows with 43 examples per class. It includes controlled paraphrases based on a small number of original minority-class summaries.
- `Methodology & Limits` explains the balancing process and the limits of the data and evaluation.

The rephrased examples are not independent customer observations. Model scores are exploratory and do not establish performance on new customer reviews. There is no separate validation set. Possible-sensitivity labels are review categories, not confirmed medical events or diagnoses. The app does not provide medical advice.

## Report

Open the report: `reports/Shineup_Consumer_Feedback_Project_Report_EN.pdf`.

## Presentation

Open the presentation: `reports/Shineup_Consumer_Feedback_Project_Presentation_EN.pptx`.
 

## Run locally

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The workbook in `data/` loads automatically. The app uses the `Original Data` sheet for visual summaries and the `Balanced Data` sheet to train the prototype classifier.

## Run from GitHub with Streamlit Community Cloud

The Streamlit app can be deployed from a private GitHub repository. Select the repository's `main` branch and set the app file to `cosmetics_consumer_health/streamlit_app.py`. Keep the workbook out of GitHub. After deployment, upload the English workbook through the app's sidebar if the deployed version does not include it.

An Excel file uploaded to the cloud app is sent to the Streamlit server for analysis and classification. Upload only data you are authorized to share. Streamlit Community Cloud deployment instructions are in the [official deploy guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy); app visibility follows the connected repository unless changed in app settings, as described in the [sharing guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## Model and evaluation

The prototype uses TF-IDF text features and logistic regression. It makes a stratified 75/25 train/test split of the balanced sheet, then fits the interactive classifier on all balanced rows. The 43-row holdout is not an independent real-world test because the balanced sheet contains paraphrases derived from a small original sample. The reported metrics should be read as a demonstration of the workflow only.

The classes are `normal`, `product_issue`, `possible_sensitivity`, and `human_review`. Predictions in the last two classes, and low-confidence predictions, are flagged for human review. Model confidence is not calibrated.
