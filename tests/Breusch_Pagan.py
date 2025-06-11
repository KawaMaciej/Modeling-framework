
from numpy.typing import NDArray
from typing import Any
import numpy as np
from scipy.stats import chi2

class Breusch_Pagan:
    def __init__(self, X, Y, model) -> None:
        self.X_original: NDArray = X.copy()
        self.Y: NDArray = Y
        self.model: Any = model
        self.bp_stat: float = np.nan
        self.p_value: float = np.nan
    def run(self):
        n = self.X_original.shape[0]

        residuals = self.Y - self.model.predict(self.X_original)

        sq_residuals = residuals ** 2

        X_aux = np.column_stack((np.ones(n), self.X_original))

        beta_aux = np.linalg.pinv(X_aux.T @ X_aux) @ X_aux.T @ sq_residuals
        sq_residuals_pred = X_aux @ beta_aux

        ssr = np.sum((sq_residuals_pred - sq_residuals.mean()) ** 2)
        sst = np.sum((sq_residuals - sq_residuals.mean()) ** 2)
        R_squared = ssr / sst

        self.bp_stat = n * R_squared

        df = X_aux.shape[1] - 1

        self.p_value = float(1 - chi2.cdf(self.bp_stat, df))

        return self.bp_stat, self.p_value
