import numpy as np
from typing import Type, List, Any


class AdaBoostClassifier:
    """
    AdaBoost classifier using a generic weak learner class.

    Parameters
    ----------
    model_class : Type
        A class implementing `fit(X, y)` and `predict(X)` methods.
    n_estimators : int
        Number of weak learners to train.
    verbose : bool, optional
        Whether to print progress and warnings during training.
    """
    def __init__(self,
                 model_class: Any,
                 n_estimators: int,
                 verbose: bool = True) -> None:
        self.model_class: Type = model_class
        self.n_estimators: int = n_estimators
        self.models: List[Any] = []
        self.alphas: List[float] = []
        self.verbose: bool = verbose

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the AdaBoost classifier.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        Y : np.ndarray
            Binary class labels (0 or 1), shape (n_samples,).
        """
        Y = 2 * Y - 1  
        w = np.ones(X.shape[0]) / X.shape[0]

        for i in range(self.n_estimators):
            model = self.model_class()
            model.fit(X, ((Y + 1) // 2))
            pred = model.predict(X)
            pred = 2 * pred - 1 

            err = np.sum(w * (pred != Y)) / np.sum(w)

            if err == 0:
                alpha = 10
            elif err >= 0.5:
                if self.verbose:
                    print(f"⚠️ Estimator {i}: weak learner has error {err:.4f} (≥ 0.5). Skipping.")
                continue
            else:
                alpha = 0.5 * np.log((1 - err) / err)

            self.models.append(model)
            self.alphas.append(alpha)

            w *= np.exp(-alpha * Y * pred)
            w /= np.sum(w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels in {-1, 1}.
        """
        pred = sum(alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.models))
        return np.sign(pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the weighted sum of weak learner predictions.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        
        np.ndarray
            Weighted sum of predictions.
        """
        raw_score = sum(alpha * (2 * clf.predict(X) - 1) 
                        for alpha, clf in zip(self.alphas, self.models))
        
        probs_pos = 1 / (1 + np.exp(-2 * raw_score)) 
        
        return np.column_stack([1 - probs_pos, probs_pos])
    




class AdaBoostRegressor:
    """
    AdaBoost regressor supporting multiple loss functions.

    Parameters
    ----------
    model_class : Type
        A regression model class implementing `fit(X, y, sample_weight)` and `predict(X)`.
    n_estimators : int, optional
        Number of weak learners to train (default=50).
    learning_rate : float, optional
        Scaling factor for model contributions (default=1.0).
    error_threshold : float, optional
        Maximum acceptable error for a weak learner (default=0.9).
    verbose : bool, optional
        Whether to print progress and warnings (default=True).
    loss_type : str, optional
        Loss function type: 'linear', 'square', or 'exponential'.
    """
    def __init__(self,
                 model_class: Any,
                 n_estimators: int = 50,
                 learning_rate: float = 1.0,
                 error_threshold: float = 0.9,
                 verbose: bool = True,
                 loss_type: str = 'linear') -> None:
        self.model_class: Type = model_class
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.error_threshold: float = error_threshold
        self.models: List[Any] = []
        self.alphas: List[float] = []
        self.verbose: bool = verbose
        self.loss_type: str = loss_type

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute normalized loss based on the selected loss type.

        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            Normalized loss for each sample.
        """
        error = np.abs(y_true - y_pred)
        max_error = np.max(error)

        if max_error == 0:
            return np.zeros_like(error)

        normalized_error = error / max_error

        if self.loss_type == 'linear':
            return normalized_error
        elif self.loss_type == 'square':
            return normalized_error ** 2
        elif self.loss_type == 'exponential':
            return 1 - np.exp(-normalized_error)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the AdaBoost regressor.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Target values, shape (n_samples,).
        """
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        successful_models = 0

        for i in range(self.n_estimators):
            try:
                model = self.model_class()
                model.fit(X, y, sample_weight=w)
                pred = model.predict(X)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Estimator {i}: Failed to train - {str(e)}")
                continue

            loss = self._compute_loss(y, pred)

            if np.max(loss) == 0:
                if self.verbose:
                    print(f"✓ Estimator {i}: Perfect fit achieved!")
                self.models.append(model)
                self.alphas.append(self.learning_rate)
                successful_models += 1
                break

            error = np.sum(w * loss)

            if error >= self.error_threshold:
                if self.verbose:
                    print(f"⚠️ Estimator {i}: weak learner has error {error:.4f} (≥ {self.error_threshold}). Skipping.")
                continue

            error = min(max(error, 1e-10), 1 - 1e-10)

            beta = error / (1 - error)
            alpha = self.learning_rate * np.log(1 / beta)

            self.models.append(model)
            self.alphas.append(alpha)
            successful_models += 1

            w *= np.power(beta, 1 - loss)
            w_sum = np.sum(w)

            if w_sum <= 1e-10:
                if self.verbose:
                    print(f"⚠️ Estimator {i}: Weights collapsed to zero. Stopping early.")
                break

            w /= w_sum

            if self.verbose:
                print(f"✓ Estimator {i}: error={error:.4f}, alpha={alpha:.4f}")

        if successful_models == 0:
            if self.verbose:
                print("⚠️ No successful AdaBoost models. Falling back to single base model.")
            try:
                fallback_model = self.model_class()
                fallback_model.fit(X, y)
                self.models = [fallback_model]
                self.alphas = [1.0]
            except Exception as e:
                raise ValueError(f"Failed to train even a single base model: {str(e)}")

        if self.verbose:
            print(f"✓ Training complete. {len(self.models)} models trained successfully.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if not self.models:
            raise ValueError("No models trained. Call fit() first.")

        if len(self.models) == 1:
            return self.models[0].predict(X)

        predictions = np.array([model.predict(X) for model in self.models])
        weights = np.array(self.alphas)

        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred