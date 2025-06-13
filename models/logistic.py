import numpy as np
from numpy.typing import NDArray
from metrics.classification_metrics import *


class LogisticRegression:
    """
    Multiclass Logistic Regression classifier with optional L1 or L2 regularization.

    This implementation uses softmax for multiclass classification and supports both
    L1 and L2 regularization for weight penalization. Bias is handled via an extra feature column of ones.

    Attributes:
        n_features (int): Number of input features (excluding bias).
        n_classes (int): Number of target classes.
        n_iter (int): Number of training iterations.
        lr (float): Learning rate for gradient descent.
        regularization (str): Regularization type: 'None', 'l1', or 'l2'.
        alpha (float): Regularization strength (used if L1 or L2 is selected).
        theta (NDArray): Weight matrix of shape (n_features + 1, n_classes), including bias.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_iter: int = 1000,
        lr: float = 0.001,
        regularization: str = "None",
        alpha: float = 0.2
    ) -> None:
        """
        Initialize the Logistic Regression model.

        Args:
            n_features (int): Number of input features (excluding bias).
            n_classes (int): Number of output classes.
            n_iter (int, optional): Number of training iterations. Defaults to 1000.
            lr (float, optional): Learning rate. Defaults to 0.001.
            regularization (str, optional): Type of regularization ('None', 'l1', 'l2'). Defaults to 'None'.
            alpha (float, optional): Regularization strength. Defaults to 0.2.
        """
        valid_regularizations = {"None", "l1", "l2"}
        if regularization not in valid_regularizations:
            raise ValueError(f"regularization must be one of {valid_regularizations}, got '{regularization}'")
        
        self.theta: NDArray = np.zeros((n_features + 1, n_classes))  # +1 for bias
        self.n_iter = n_iter
        self.lr = lr
        self.n_classes = n_classes
        self.n_features = n_features
        self.regularization = regularization
        self.alpha = alpha if regularization in {"l1", "l2"} else 0.0

    def score_function(self, x: NDArray, k: int) -> float:
        """
        Compute the raw score (logit) for class k given input x.

        Args:
            x (NDArray): Feature vector including bias term.
            k (int): Class index.

        Returns:
            float: Logit score for class k.
        """
        return float(np.dot(x, self.theta[:, k]))

    def softmax(self, X: NDArray) -> NDArray:
        """
        Compute softmax probabilities for all classes.

        Args:
            X (NDArray): Feature matrix of shape (n_samples, n_features + 1).

        Returns:
            NDArray: Probability matrix of shape (n_samples, n_classes).
        """
        logits = X @ self.theta
        logits -= np.max(logits, axis=1, keepdims=True)  # Numerical stability
        exp_scores = np.exp(logits)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def cross_entropy(self, X: NDArray) -> float:
        """
        (Optional) Compute the cross-entropy loss.
        Placeholder for future implementation.

        Args:
            X (NDArray): Input feature matrix.

        Returns:
            float: Cross-entropy loss.
        """
        pass  # Not implemented yet

    def fit(self, X: NDArray, Y: NDArray) -> "LogisticRegression":
        """
        Train the logistic regression model using gradient descent.

        Args:
            X (NDArray): Input features of shape (n_samples, n_features).
            Y (NDArray): Integer class labels of shape (n_samples,).

        Returns:
            LogisticRegression: The trained model instance.
        """
        Y_onehot = np.eye(self.n_classes)[Y]
        X = self._add_bias(X)

        for _ in range(self.n_iter):
            probs = self.softmax(X)
            error = probs - Y_onehot
            grad = X.T @ error / X.shape[0]

            if self.regularization == "l2":
                grad += self.alpha * self.theta  # L2 penalty

            if self.regularization == "l1":
                reg_term = np.sign(self.theta)
                reg_term[0, :] = 0  # No bias regularization
                grad += self.alpha * reg_term

            self.theta -= self.lr * grad

        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities for input data.

        Args:
            X (NDArray): Input features of shape (n_samples, n_features).

        Returns:
            NDArray: Predicted class probabilities of shape (n_samples, n_classes).
        """
        X = self._add_bias(X)
        return self.softmax(X)

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class labels for input data.

        Args:
            X (NDArray): Input features of shape (n_samples, n_features).

        Returns:
            NDArray: Predicted class labels of shape (n_samples,).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _add_bias(self, X: NDArray) -> NDArray:
        """
        Add a column of ones to X to account for the bias term.

        Args:
            X (NDArray): Feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray: Modified matrix of shape (n_samples, n_features + 1).
        """
        return np.hstack([np.ones((X.shape[0], 1)), X])


