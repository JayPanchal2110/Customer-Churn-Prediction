"""Educational binary logistic regression implemented using NumPy only."""
from __future__ import annotations
import numpy as np


class CustomLogisticRegression:
    """Binary logistic regression trained with batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Gradient descent step size.
    n_iterations : int
        Maximum number of gradient updates.
    class_weight : None or "balanced"
        If "balanced", gives greater weight to the less frequent class.
    l2 : float
        L2 regularization strength (intercept is not regularized).
    """

    def __init__(self, learning_rate=0.05, n_iterations=3000,
                 class_weight=None, l2=0.001, tolerance=1e-8):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.class_weight = class_weight
        self.l2 = float(l2)
        self.tolerance = float(tolerance)
        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []

    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid, returning values between 0 and 1."""
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z)
        positive = z >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1.0 + exp_z)
        return result

    @staticmethod
    def log_loss(y_true, probabilities, sample_weight=None):
        """Binary cross-entropy with clipping for numerical stability."""
        y = np.asarray(y_true, dtype=float).reshape(-1)
        p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-15, 1 - 1e-15)
        losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        if sample_weight is None:
            return float(np.mean(losses))
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        return float(np.sum(w * losses) / np.sum(w))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if len(y) == 0:
            raise ValueError("Training data cannot be empty")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only 0 and 1")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values; preprocess first")

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0

        sample_weights = np.ones(n_samples, dtype=float)
        if self.class_weight == "balanced":
            counts = np.bincount(y.astype(int), minlength=2)
            if np.any(counts == 0):
                raise ValueError("Balanced class weighting requires both classes")
            sample_weights = np.where(
                y == 1, n_samples / (2.0 * counts[1]),
                n_samples / (2.0 * counts[0])
            )

        total_weight = np.sum(sample_weights)
        self.loss_history_ = []
        previous_loss = np.inf

        for _ in range(self.n_iterations):
            probabilities = self.sigmoid(X @ self.weights_ + self.bias_)
            errors = (probabilities - y) * sample_weights
            grad_w = (X.T @ errors) / total_weight + self.l2 * self.weights_
            grad_b = float(np.sum(errors) / total_weight)

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            updated_probabilities = self.sigmoid(X @ self.weights_ + self.bias_)
            loss = self.log_loss(y, updated_probabilities, sample_weights)
            loss += 0.5 * self.l2 * float(np.sum(self.weights_ ** 2))
            self.loss_history_.append(loss)

            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        return self

    def predict_proba(self, X):
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted")
        X = np.asarray(X, dtype=float)
        probabilities = self.sigmoid(X @ self.weights_ + self.bias_)
        return np.column_stack((1.0 - probabilities, probabilities))

    def predict(self, X, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
