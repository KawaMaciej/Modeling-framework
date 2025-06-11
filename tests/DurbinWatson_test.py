from numpy.typing import NDArray
from typing import Any, Tuple
import numpy as np


class DurbinWatson:
    def __init__(self, X: NDArray, Y: NDArray, model: Any ) -> None:
        self.model = model
        self.X_original = X.copy()
        self.Y = Y

    def run(self) -> None:
        resid = self.model.resid(self.X_original, self.Y)
        resid_lag = np.empty_like(resid, dtype=float)
        resid_lag[0] = resid[0]
        resid_lag[1:] = resid[:-1]
        d = np.sum((resid[1:] - resid[:-1])**2) / np.sum(resid**2)

        return d 
