import numpy as np
from scipy.stats import mode
from typing import Any, Optional
import copy

class BaggingClassifier:
    """
    A simple Bagging classifier ensemble.

    Parameters:
    -----------
    estimator : object
        A classifier instance with fit and predict (and optionally predict_proba) methods.
    n_splits : int, default=4
        Number of bootstrap samples/models to train.
    soft_voting : bool, default=False
        If True, use soft voting by averaging predicted probabilities.
        If False, use hard voting (majority class).
    replacement : bool, default=True
        Whether sampling is done with replacement.
    random_state : Optional[int], default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        estimator: Any,
        n_splits: int = 4,
        soft_voting: bool = False,
        replacement: bool = True,
        random_state: Optional[int] = 42
    ) -> None:
        self.n_splits = n_splits
        self.estimator = estimator
        self.soft_voting = soft_voting
        self.replacement = replacement
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BaggingClassifier":
        """
        Fit the ensemble of estimators on bootstrap samples.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        Y : array-like of shape (n_samples,)
            Target values.

        Returns:
        --------
        self : BaggingClassifier
            Fitted estimator.
        """
        self.models = []
        X = np.array(X)
        Y = np.array(Y)

        for _ in range(self.n_splits):
            indices = np.random.choice(len(X), size=len(X), replace=self.replacement)
            X_sample = X[indices]
            Y_sample = Y[indices]

            model = clone_estimator(self.estimator)
            model.fit(X_sample, Y_sample)
            self.models.append(model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.array(X)
        if not self.soft_voting:
            predictions = np.array([model.predict(X) for model in self.models])
            majority_votes, _ = mode(predictions, axis=0, keepdims=False)
            return majority_votes
        else:
            probas = np.array([model.predict_proba(X) for model in self.models])
            avg_probas = np.mean(probas, axis=0)
            return np.argmax(avg_probas, axis=1)


class BaggingRegressor:
    """
    A simple Bagging regressor ensemble.

    Parameters:
    -----------
    estimator : callable
        A regression estimator class with fit and predict methods.
    n_splits : int, default=4
        Number of bootstrap samples/models to train.
    replacement : bool, default=True
        Whether sampling is done with replacement.
    random_state : Optional[int], default=42
        Random seed for reproducibility.
    model_args : dict, optional
        Additional arguments to pass to the estimator constructor.
    """

    def __init__(
        self,
        estimator: Any,
        n_splits: int = 4,
        replacement: bool = True,
        random_state: Optional[int] = 42,
        **model_args: Any
    ) -> None:
        self.n_splits = n_splits
        self.model_args = model_args
        self.estimator = estimator
        self.replacement = replacement
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BaggingRegressor":
        """
        Fit the ensemble of regressors on bootstrap samples.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        Y : array-like of shape (n_samples,)
            Target values.

        Returns:
        --------
        self : BaggingRegressor
            Fitted estimator.
        """
        self.models = []
        X = np.array(X)
        Y = np.array(Y)

        for _ in range(self.n_splits):
            indices = np.random.choice(len(X), size=len(X), replace=self.replacement)
            X_sample = X[indices]
            Y_sample = Y[indices]

            model = self.estimator(**self.model_args)
            model.fit(X_sample, Y_sample)
            self.models.append(model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression targets for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns:
        --------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.array(X)
        preds = np.array([model.predict(X) for model in self.models])
        avg_preds = np.mean(preds, axis=0)
        return avg_preds


def clone_estimator(estimator: Any) -> Any:
    """
    Create a deep copy of the given estimator.

    Parameters:
    -----------
    estimator : Any
        The estimator instance to clone.

    Returns:
    --------
    Any
        A deep copy (clone) of the estimator.
    """
    return copy.deepcopy(estimator)