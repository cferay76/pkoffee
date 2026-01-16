import numpy as np
import pytest

from pkoffee.metrics import compute_r2


def test_compute_r2_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    r2 = compute_r2(y_true, y_pred)

    assert r2 == 1.0


def test_compute_r2_docstring_example():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9])

    r2 = compute_r2(y_true, y_pred)

    assert np.isclose(r2, 0.9920, atol=1e-4)


def test_compute_r2_size_mismatch():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        compute_r2(y_true, y_pred)
