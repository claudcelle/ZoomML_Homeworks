"""Model evaluation metrics."""

import numpy as np


def RMSE(y, y0):
    assert y.shape == y0.shape, (
        f"Shape diverse: y={y.shape}, y_0={y0.shape}"
    )
    return np.sqrt(np.mean((y - y0) * (y - y0)))
