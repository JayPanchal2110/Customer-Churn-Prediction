
"""Train custom NumPy logistic regression and benchmark against Scikit-learn."""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

from src.data_utils import PROJECT_ROOT, load_dataset, build_preprocessor
from src.logistic_regression import CustomLogisticRegression
from src.evaluation import evaluate_binary_classifier


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50


def check_processed_data(name, data):
    """Check that processed model inputs are finite and valid."""

    data = np.asarray(data, dtype=float)

    print(f"\nChecking {name}...")
    print(f"Shape: {data.shape}")

    if data.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.isfinite(data).all():
        raise ValueError(
            f"{name} contains NaN or infinite values. "
            "Inspect the preprocessing pipeline."
        )

    print(f"All values finite: {np.isfinite(data).all()}")
    print(f"Minimum value: {data.min():.6f}")
    print(f"Maximum value: {data.max():.6f}")

    return data


def save_confusion_matrix(cm, title, output_path):
    """Save a confusion matrix plot."""

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted stayed", "Predicted churned"],
        yticklabels=["Actual stayed", "Actual churned"],
    )

    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    # -----------------------------------------------------
    # 1. Prepare project output directories
    # -----------------------------------------------------

    outputs = PROJECT_ROOT / "outputs"
    models_dir = PROJECT_ROOT / "models"

    outputs.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 2. Load dataset
    # -----------------------------------------------------

    X, y, df = load_dataset()

    print("=" * 60)
    print("CUSTOM LOGISTIC REGRESSION FOR CHURN PREDICTION")
    print("=" * 60)

    print(f"\nLoaded {len(df):,} customer records.")
    print(f"Number of predictor columns: {X.shape[1]}")

    print("\nTarget distribution:")
    print(
        y.value_counts()
        .rename(index={0: "Stayed (0)", 1: "Churned (1)"})
    )

    print(f"\nOverall churn rate: {y.mean():.2%}")

    # -----------------------------------------------------
    # 3. Initial exploratory data analysis
    # -----------------------------------------------------

    print("\nGenerating initial EDA outputs...")

    df.describe(include="all").transpose().to_csv(
        outputs / "data_summary.csv"
    )

    plt.figure(figsize=(7, 4))

    sns.countplot(
        x=y.map({0: "Stayed", 1: "Churned"})
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Customer outcome")
    plt.ylabel("Number of customers")
    plt.tight_layout()

    plt.savefig(
        outputs / "churn_distribution.png",
        dpi=150
    )

    plt.close()

    # -----------------------------------------------------
    # 4. Split data into training and testing sets
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")

    # -----------------------------------------------------
    # 5. Fit preprocessing on training data only
    # -----------------------------------------------------

    preprocessor, _, _ = build_preprocessor(X_train)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Convert sparse one-hot encoded matrices to dense arrays
    # because the educational NumPy model uses dense arrays.

    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()

    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    # IMPORTANT:
    # These checks must occur AFTER X_train_t and X_test_t
    # have been created.

    X_train_t = check_processed_data(
        "X_train_t",
        X_train_t
    )

    X_test_t = check_processed_data(
        "X_test_t",
        X_test_t
    )

    if X_train_t.shape[1] != X_test_t.shape[1]:
        raise ValueError(
            "Training and testing feature counts do not match."
        )

    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    if not np.isfinite(y_train_array).all():
        raise ValueError("Training target contains invalid values.")

    if not np.isfinite(y_test_array).all():
        raise ValueError("Testing target contains invalid values.")

    # -----------------------------------------------------
    # 6. Train custom NumPy logistic regression
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING CUSTOM NUMPY LOGISTIC REGRESSION")
    print("=" * 60)

    custom_model = CustomLogisticRegression(
        learning_rate=0.005,
        n_iterations=10000,
        class_weight="balanced",
        l2=0.001,
    )

    custom_model.fit(
        X_train_t,
        y_train_array
    )

    if (
        not np.isfinite(custom_model.weights_).all()
        or not np.isfinite(custom_model.bias_)
    ):
        raise FloatingPointError(
            "Custom model learned invalid parameters. "
            "Check the learning rate and input data."
        )

    if (
        not custom_model.loss_history_
        or not np.isfinite(custom_model.loss_history_).all()
    ):
        raise FloatingPointError(
            "Custom model produced invalid training loss values."
        )

    print("Custom model training completed.")
    print(f"Training iterations: {len(custom_model.loss_history_)}")
    print(f"Final training loss: {custom_model.loss_history_[-1]:.6f}")

    custom_probabilities = (
        custom_model.predict_proba(X_test_t)[:, 1]
    )

    print("\n--- Numerical diagnostics ---")

    print("Training data finite:", np.isfinite(X_train_t).all())
    print("Testing data finite:", np.isfinite(X_test_t).all())

    print(
        "Custom weights finite:",
        np.isfinite(custom_model.weights_).all()
    )

    print(
        "Maximum absolute weight:",
        np.max(np.abs(custom_model.weights_))
    )

    print(
        "Custom probabilities finite:",
        np.isfinite(custom_probabilities).all()
    )

    print(
        "Custom probability range:",
        np.min(custom_probabilities),
        np.max(custom_probabilities)
    )

    if not np.isfinite(custom_probabilities).all():
        raise FloatingPointError(
            "Custom model produced invalid prediction probabilities."
        )

    
    # -----------------------------------------------------
    # 7. Train Scikit-learn benchmark
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING SCIKIT-LEARN BENCHMARK")
    print("=" * 60)

    sklearn_model = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    # Validate the input data before training.
    print("\n--- Before scikit-learn training ---")
    print("X_train shape:", X_train_t.shape)
    print("X_train dtype:", X_train_t.dtype)
    print("X_train finite:", np.isfinite(X_train_t).all())
    print("X_train max abs:", np.max(np.abs(X_train_t)))
    print("X_train mean abs:", np.mean(np.abs(X_train_t)))
    print("X_train contiguous:", X_train_t.flags["C_CONTIGUOUS"])
    print("Target values:", np.unique(y_train_array))
    print("Target dtype:", y_train_array.dtype)

    if not np.isfinite(X_train_t).all():
        raise ValueError("Training features contain NaN or infinity.")

    if not np.isfinite(X_test_t).all():
        raise ValueError("Testing features contain NaN or infinity.")

    # Fit the scikit-learn model.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        sklearn_model.fit(
            X_train_t,
            y_train_array,
        )

    # Report convergence warnings, if any.
    convergence_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]

    if convergence_warnings:
        for warning in convergence_warnings:
            print(
                "Scikit-learn convergence warning:",
                warning.message,
            )

    # Generate probabilities after successful fitting.
    sklearn_probabilities = (
        sklearn_model.predict_proba(X_test_t)[:, 1]
    )

    if not np.isfinite(sklearn_probabilities).all():
        raise FloatingPointError(
            "Scikit-learn produced invalid prediction probabilities."
        )

    print("Scikit-learn benchmark training completed.")
    print("Scikit-learn weights finite:",
          np.isfinite(sklearn_model.coef_).all())
    print("Scikit-learn probabilities finite:",
          np.isfinite(sklearn_probabilities).all())

    # -----------------------------------------------------
    # 8. Evaluate both models
    # -----------------------------------------------------

    custom_metrics, custom_cm, _ = evaluate_binary_classifier(
        y_test_array,
        custom_probabilities,
        THRESHOLD,
    )

    sklearn_metrics, sklearn_cm, _ = evaluate_binary_classifier(
        y_test_array,
        sklearn_probabilities,
        THRESHOLD,
    )

    results = {
        "Custom NumPy Logistic Regression": custom_metrics,
        "Scikit-learn Logistic Regression": sklearn_metrics,
    }

    results_df = pd.DataFrame(results).T

    if not np.isfinite(results_df.to_numpy(dtype=float)).all():
        raise FloatingPointError(
            "Evaluation contains invalid metric values."
        )

    results_df.to_csv(
        outputs / "model_comparison.csv"
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"Classification threshold: {THRESHOLD:.2f}")
    print(results_df.round(4).to_string())

    # -----------------------------------------------------
    # 9. Save confusion matrices
    # -----------------------------------------------------

    save_confusion_matrix(
        custom_cm,
        "Custom NumPy Logistic Regression",
        outputs / "custom_confusion_matrix.png",
    )

    save_confusion_matrix(
        sklearn_cm,
        "Scikit-learn Logistic Regression",
        outputs / "sklearn_confusion_matrix.png",
    )

    # -----------------------------------------------------
    # 10. Save custom model training loss
    # -----------------------------------------------------

    plt.figure(figsize=(8, 4))

    plt.plot(
        range(1, len(custom_model.loss_history_) + 1),
        custom_model.loss_history_,
    )

    plt.title("Custom Model Training Loss")
    plt.xlabel("Training iteration")
    plt.ylabel("Weighted log-loss + regularization")
    plt.tight_layout()

    plt.savefig(
        outputs / "training_loss.png",
        dpi=150
    )

    plt.close()

    # -----------------------------------------------------
    # 11. Save ROC curve comparison
    # -----------------------------------------------------

    plt.figure(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        y_test_array,
        custom_probabilities,
        name="Custom NumPy model",
    )

    RocCurveDisplay.from_predictions(
        y_test_array,
        sklearn_probabilities,
        name="Scikit-learn model",
    )

    plt.title("ROC Curve Comparison")
    plt.tight_layout()

    plt.savefig(
        outputs / "roc_curve_comparison.png",
        dpi=150
    )

    plt.close()

    # -----------------------------------------------------
    # 12. Save trained model bundle
    # -----------------------------------------------------

    bundle = {
        "preprocessor": preprocessor,
        "custom_model": custom_model,
        "sklearn_model": sklearn_model,
        "feature_columns": X.columns.tolist(),
        "threshold": THRESHOLD,
        "dataset_name": "IBM Telco Customer Churn",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    model_path = models_dir / "churn_model_bundle.joblib"

    joblib.dump(
        bundle,
        model_path
    )

    # -----------------------------------------------------
    # 13. Save run metadata and metrics
    # -----------------------------------------------------

    metadata = {
        "dataset": "IBM Telco Customer Churn",
        "rows": int(len(df)),
        "predictor_columns": int(X.shape[1]),
        "churn_rate": float(y.mean()),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": THRESHOLD,
        "custom_training_iterations": len(
            custom_model.loss_history_
        ),
        "custom_final_training_loss": float(
            custom_model.loss_history_[-1]
        ),
        "metrics": {
            model_name: {
                metric_name: float(metric_value)
                for metric_name, metric_value in metrics.items()
            }
            for model_name, metrics in results.items()
        },
        "notes": (
            "Metrics were calculated from this run's held-out test set. "
            "Interpret results using the documented split and threshold."
        ),
    }

    with open(
        outputs / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    # -----------------------------------------------------
    # 14. Final status
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nSaved model bundle:\n{model_path}")

    print(f"\nSaved evaluation outputs:\n{outputs}")

    print("\nGenerated files:")
    for output_file in sorted(outputs.iterdir()):
        if output_file.is_file():
            print(f"  - {output_file.name}")

    print("\nNext step:")
    print("Run: ./.venv/bin/python -m streamlit run app.py")


if __name__ == "__main__":
    main()
