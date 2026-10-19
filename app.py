"""Streamlit UI for the trained churn model bundle."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "churn_model_bundle.joblib"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction")
st.caption("Custom NumPy logistic regression · IBM Telco Customer Churn")

if not BUNDLE_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.markdown(
        "1. Download the CSV described in `data/README.md`.\n"
        "2. Save it as `data/raw/Telco-Customer-Churn.csv`.\n"
        "3. Install dependencies and run `python -m src.train`.\n"
        "4. Restart this app."
    )
    st.stop()

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

bundle = load_bundle(BUNDLE_PATH)
preprocessor = bundle["preprocessor"]
model = bundle["custom_model"]
feature_columns = bundle["feature_columns"]
threshold_default = float(bundle.get("threshold", 0.5))

st.info("Enter a plausible customer profile. Predictions are estimates, not guarantees.")
with st.form("customer_form"):
    values = {}
    cols = st.columns(2)
    # The form adapts to the common IBM Telco schema. Defaults are examples only.
    for i, feature in enumerate(feature_columns):
        with cols[i % 2]:
            if feature == "gender":
                values[feature] = st.selectbox(feature, ["Female", "Male"])
            elif feature == "SeniorCitizen":
                values[feature] = st.selectbox(feature, [0, 1], format_func=lambda x: "Yes" if x else "No")
            elif feature == "tenure":
                values[feature] = st.slider(feature, 0, 72, 12)
            elif feature == "MonthlyCharges":
                values[feature] = st.number_input(feature, min_value=0.0, max_value=300.0, value=70.0, step=1.0)
            elif feature == "TotalCharges":
                values[feature] = st.number_input(feature, min_value=0.0, max_value=20000.0, value=840.0, step=10.0)
            elif feature in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
                values[feature] = st.selectbox(feature, ["No", "Yes"])
            elif feature in ("MultipleLines", "InternetService", "OnlineSecurity",
                             "OnlineBackup", "DeviceProtection", "TechSupport",
                             "StreamingTV", "StreamingMovies"):
                choices = {
                    "MultipleLines": ["No", "Yes", "No phone service"],
                    "InternetService": ["DSL", "Fiber optic", "No"],
                    "OnlineSecurity": ["No", "Yes", "No internet service"],
                    "OnlineBackup": ["No", "Yes", "No internet service"],
                    "DeviceProtection": ["No", "Yes", "No internet service"],
                    "TechSupport": ["No", "Yes", "No internet service"],
                    "StreamingTV": ["No", "Yes", "No internet service"],
                    "StreamingMovies": ["No", "Yes", "No internet service"],
                }
                values[feature] = st.selectbox(feature, choices[feature])
            elif feature == "Contract":
                values[feature] = st.selectbox(feature, ["Month-to-month", "One year", "Two year"])
            elif feature == "PaperlessBilling":
                values[feature] = st.selectbox(feature, ["No", "Yes"])
            elif feature == "PaymentMethod":
                values[feature] = st.selectbox(feature, [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ])
            else:
                # Fallback for any unexpected column: use a safe categorical choice.
                values[feature] = st.text_input(feature, value="")

    threshold = st.slider("Churn classification threshold", 0.05, 0.95, threshold_default, 0.05)
    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    row = pd.DataFrame([{col: values[col] for col in feature_columns}], columns=feature_columns)
    # Match common Telco data types before preprocessing.
    for col in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    probability = float(model.predict_proba(transformed)[0, 1])
    prediction = int(probability >= threshold)
    risk = "High risk" if probability >= 0.65 else ("Medium risk" if probability >= 0.35 else "Low risk")

    a, b, c = st.columns(3)
    a.metric("Estimated churn probability", f"{probability:.1%}")
    b.metric("Prediction", "Likely to churn" if prediction else "Likely to stay")
    c.metric("Risk band", risk)
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Threshold: {threshold:.2f}. Risk-band cutoffs are illustrative UI labels, not calibrated business rules.")
    st.warning("This is a statistical estimate based on historical patterns. It should not be used as the sole basis for decisions about an individual customer.")

with st.expander("View model evaluation from training"):
    comparison_path = ROOT / "outputs" / "model_comparison.csv"
    if comparison_path.exists():
        st.dataframe(pd.read_csv(comparison_path, index_col=0), use_container_width=True)
    else:
        st.write("Run `python -m src.train` to generate evaluation results.")
