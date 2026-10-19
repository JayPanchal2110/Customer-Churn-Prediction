# Custom Logistic Regression for Customer Churn Prediction

A reproducible, laptop-friendly major project that implements binary logistic regression **from scratch with NumPy**, evaluates it, benchmarks it against Scikit-learn, and serves predictions through Streamlit.

## Project objectives

- Explore and prepare customer churn data.
- Implement sigmoid, binary cross-entropy/log-loss, gradient descent, class weighting, and probability predictions with NumPy.
- Evaluate Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix.
- Benchmark against Scikit-learn LogisticRegression using the same train/test split and preprocessing.
- Provide a small interactive Streamlit demo.
- Record reproducible results for an analytical report.

## Dataset

Recommended: IBM Telco Customer Churn CSV (7,043 rows, commonly distributed with 21 columns).

- IBM GitHub CSV: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
- Kaggle page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the CSV and save it as:

`data/raw/Telco-Customer-Churn.csv`

The raw dataset is intentionally not bundled into this ZIP. Do not commit customer data unless you have confirmed you are permitted to share it.

## Requirements

- Python 3.10–3.12 recommended
- Git
- Internet connection for installing packages and downloading the dataset
- VS Code or Jupyter (optional, but helpful)

## Quick start: macOS / Linux

```bash
cd custom-logistic-regression-churn
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start: Windows PowerShell

```powershell
cd custom-logistic-regression-churn
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

1. Put the CSV at `data/raw/Telco-Customer-Churn.csv`.
2. Train both models and create evaluation outputs:

```bash
python -m src.train
```

3. Launch the interactive app:

```bash
streamlit run app.py
```

4. Run automated tests:

```bash
python -m pytest
```

Outputs are saved under `outputs/`; the trained bundle is saved under `models/`. These generated files are ignored by Git by default.

## Project workflow

1. Data validation and EDA
2. Train/test split with stratification
3. Preprocessing fitted on training data only
4. Custom NumPy logistic regression training
5. Evaluation of the custom model
6. Scikit-learn benchmark
7. Saved model bundle and Streamlit prediction interface
8. Analytical report using measured results

## Main files

```text
src/logistic_regression.py  Custom NumPy implementation
src/data_utils.py           Dataset loading and preprocessing setup
src/train.py                Training, evaluation, artifact saving
app.py                      Streamlit application
notebooks/                   Guided notebook
tests/                       Basic model tests
report/                      Final report template
```

## Important academic notes

- `Churn` is the target. The app predicts `Yes` or `No` and reports the estimated probability.
- `customerID` is an identifier and is excluded from model features.
- The default decision threshold is 0.50. You should examine precision/recall trade-offs and justify any threshold you choose.
- Class weighting is available in the custom implementation. Compare it with an unweighted run before making claims.
- Report the results produced by your own run. Do not invent or promise an accuracy score.
- The custom model is educational. The Scikit-learn model is the benchmark, not the model used by the app.
- Do not describe feature associations as causal effects.
- This project is a decision-support prototype, not a guarantee that any individual customer will churn.

## Suggested next improvements

- Add a threshold comparison page.
- Add cross-validation and confidence intervals.
- Add model explainability and calibration checks.
- Add CI with GitHub Actions.
- Deploy the Streamlit app after local evaluation is stable.
