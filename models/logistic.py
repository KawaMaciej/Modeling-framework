import numpy as np
from numpy.typing import NDArray
from metrics.classification_metrics import *
import torch
from solvers.grad_methods import GradientDescent, LBFGS
import torch.nn.functional as F

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
        alpha: float = 0.2,
        solver: str = "GD",
        m: int = 10,
        random_state = 42
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
        self.n_classes = n_classes
        self.n_features = n_features
        
        np.random.seed(random_state)

        self.theta =np.random.randn(self.n_features + 1, self.n_classes)
       
        self.solver = solver
        valid_solvers = {"GD", "LBFGS"}
        if solver not in valid_solvers:
            raise ValueError(f"Solver must be one of {valid_solvers}, got '{solver}'")
        if solver == "GD":
            self.n_iter = n_iter
            self.lr = lr
        if solver == "LBFGS":
            self.n_iter = n_iter
            self.lr = lr
            self.m = m

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
        theta = self.theta.reshape(self.n_features + 1, self.n_classes)
        logits = X @ theta
        logits -= np.max(logits, axis=1, keepdims=True) 
        exp_scores = np.exp(logits)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


    def cross_entropy(self, theta_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss (optionally regularized).

        Args:
            theta_flat (torch.Tensor): Flattened parameter tensor.

        Returns:
            torch.Tensor: Scalar tensor representing the loss.
        """
        X = torch.tensor(self.X, dtype=torch.float64)
        Y = torch.tensor(self.Y, dtype=torch.long) 
        theta = theta_flat.view(self.n_features + 1, self.n_classes)

        logits = X @ theta 

        loss = F.cross_entropy(logits, Y)

        if self.regularization == "l2":
            loss += self.alpha / 2 * torch.sum(theta[1:, :] ** 2)
        elif self.regularization == "l1":
            loss += self.alpha * torch.sum(torch.abs(theta[1:, :]))

        return loss
        
    def fit(self, X: NDArray[np.float64], Y: NDArray[np.float64]) -> "LogisticRegression":
        """
        Train the logistic regression model using gradient descent.

        Args:
            X (NDArray): Input features of shape (n_samples, n_features).
            Y (NDArray): Integer class labels of shape (n_samples,).

        Returns:
            LogisticRegression: The trained model instance.
        """
        self.Y = Y 
        self.X = self._add_bias(X)

        if self.solver == "GD":
            self.theta = GradientDescent(self.cross_entropy, self.theta, self.lr, self.n_iter)
        elif self.solver == "LBFGS":
            self.theta = LBFGS(self.cross_entropy, self.theta, self.lr, self.n_iter, self.m)

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


