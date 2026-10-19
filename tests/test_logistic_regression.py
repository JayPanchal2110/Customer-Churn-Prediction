import numpy as np
from src.logistic_regression import CustomLogisticRegression


def test_sigmoid_range_and_midpoint():
    result = CustomLogisticRegression.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(result))
    assert result[0] >= 0.0
    assert result[1] == 0.5
    assert result[2] <= 1.0


def test_model_learns_simple_separable_data():
    X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = CustomLogisticRegression(learning_rate=0.1, n_iterations=3000, l2=0.0)
    model.fit(X, y)
    predictions = model.predict(X)
    assert (predictions == y).mean() >= 5 / 6
    assert model.predict_proba(X).shape == (6, 2)


def test_invalid_labels_raise():
    model = CustomLogisticRegression()
    try:
        model.fit(np.array([[1.0], [2.0]]), np.array([0, 2]))
    except ValueError:
        return
    raise AssertionError("Expected invalid labels to raise ValueError")
