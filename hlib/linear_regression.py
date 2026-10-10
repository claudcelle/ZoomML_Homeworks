"""Linear regression training with optional regularization and prediction."""

import numpy as np


def train_linear_regression(X, y, r=0, put_ones=False):
    if put_ones:
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
    if np.isnan(X).any():
        raise Exception("X contains NaN values")
        return

    XTX = X.T @ X
    reg = r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX + reg)
    w = XTX_inv @ X.T @ y
    return w


def predict_linear_regression(X, w):
    return (w[0] + X.dot(w[1:])).ravel()
