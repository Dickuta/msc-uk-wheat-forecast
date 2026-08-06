import numpy as np

from src.metrics import rmse, mae, diebold_mariano


def test_rmse_known_values():
    y_true = [3.0, 4.0, 5.0]
    y_pred = [3.5, 4.0, 4.5]
    assert rmse(y_true, y_pred) == np.sqrt(1.0 / 6.0)


def test_mae_known_values():
    y_true = [3.0, 4.0, 5.0]
    y_pred = [3.5, 4.0, 4.5]
    assert mae(y_true, y_pred) == 1.0 / 3.0


def test_diebold_mariano_sign_favours_better_model():
    y_true = np.array([10.0, 11.0, 12.0, 13.0])
    y_pred_good = np.array([10.1, 11.2, 11.8, 13.1])
    y_pred_bad = np.array([11.0, 10.0, 13.0, 12.0])
    stat, p = diebold_mariano(y_true, y_pred_good, y_pred_bad, loss="MSE", h=1)
    assert stat < 0
    assert 0.0 <= p <= 1.0


def test_diebold_mariano_mae_loss():
    y_true = np.array([10.0, 11.0, 12.0, 13.0])
    y_pred_good = np.array([10.1, 11.2, 11.8, 13.1])
    y_pred_bad = np.array([11.0, 10.0, 13.0, 12.0])
    stat, p = diebold_mariano(y_true, y_pred_good, y_pred_bad, loss="MAE", h=2)
    assert stat < 0
    assert 0.0 <= p <= 1.0


def test_diebold_mariano_identical_forecasts():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    stat, p = diebold_mariano(y_true, y_true, y_true, loss="MSE", h=1)
    assert stat == 0.0
    assert p == 1.0
