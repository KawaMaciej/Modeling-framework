
import numpy as np
from numpy.typing import NDArray
from scipy.stats import f
from typing import Any, Tuple


class RESET_test:
    """
    Ramsey RESET test for model specification.

    This test checks whether a linear model suffers from misspecification
    by augmenting the model with higher powers of the fitted values
    and checking if they significantly improve the model.

    Parameters:
    ----------
    X : NDArray
        The original design matrix (features).
    Y : NDArray
        The target/dependent variable.
    model : Any
        A regression model that implements `fit(X, y)` and `predict(X)`.
    power : int
        Maximum power to which fitted values will be raised (default is 2).
    """

    def __init__(self, X: NDArray, Y: NDArray, model: Any, power: int = 2) -> None:
        self.X_original: NDArray = X.copy()
        self.Y: NDArray = Y
        self.model: Any = model
        self.power: int = power
        self.F_stat: float = np.nan
        self.p_value: float = np.nan

    def run(self) -> Tuple[float, float]:
        """
        Runs the RESET test.

        Returns:
        -------
        Tuple[float, float]
            A tuple containing the F-statistic and the corresponding p-value.
        """
        base_model = self.model
        base_model.fit(self.X_original, self.Y)
        fitted = base_model.predict(self.X_original)

        X_aug = self.X_original.copy()
        for i in range(2, self.power + 1):
            X_aug = np.column_stack((X_aug, fitted ** i))

        model_aug = self._clone_model()
        model_aug.fit(X_aug, self.Y)

        resid_orig = self.Y - base_model.predict(self.X_original)
        resid_aug = self.Y - model_aug.predict(X_aug)

        rss_orig = np.sum(resid_orig ** 2)
        rss_aug = np.sum(resid_aug ** 2)

        q = self.power - 1                    
        n = self.Y.shape[0]                    
        k_aug = X_aug.shape[1]                

        numerator = (rss_orig - rss_aug) / q
        denominator = rss_aug / (n - k_aug)
        F_stat = numerator / denominator

        p_value = 1 - f.cdf(F_stat, q, n - k_aug)

        # Save results
        self.F_stat = float(F_stat)
        self.p_value = float(p_value)

        return self.F_stat, self.p_value

    def _clone_model(self) -> Any:
        """
        Attempts to create a fresh instance of the provided model
        using its current parameters.

        Returns:
        -------
        Any
            A new instance of the same model.
        """
        try:
            return type(self.model)(**self.model.get_params())
        except Exception:
            return type(self.model)()