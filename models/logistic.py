import numpy as np
from numpy.typing import NDArray
from metrics.classification_metrics import *
import torch
from solvers.grad_methods import GradientDescent, LBFGS
import torch.nn.functional as F
import matplotlib.pyplot as plt
class LogisticRegression:
    """
    Multiclass Logistic Regression model supporting L1, L2, and Elastic Net regularization.

    This implementation uses softmax activation and cross-entropy loss for multiclass classification.
    It includes support for two optimization methods: Gradient Descent and L-BFGS.
    Bias is handled explicitly via feature augmentation.

    Parameters:
        n_features (int): Number of input features (excluding bias).
        n_classes (int): Number of target classes.
        n_iter (int): Maximum number of training iterations.
        lr (float): Learning rate.
        regularization (str): One of {'None', 'l1', 'l2', 'elastic_net'}.
        l1_ratio (float): L1 regularization strength.
        l2_ratio (float): L2 regularization strength.
        solver (str): Optimization algorithm, either 'GD' or 'LBFGS'.
        m (int): History size for LBFGS.
        random_state (int): Seed for reproducibility.
        tol (float): Tolerance for convergence (Δ loss threshold).

    Attributes:
        theta (NDArray[np.float64]): Weight matrix of shape (n_features + 1, n_classes).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_iter: int = 1000,
        lr: float = 0.001,
        regularization: str = "None",
        l1_ratio: float = 0.5,
        l2_ratio: float = 0.5,
        solver: str = "GD",
        m: int = 10,
        random_state: int = 0,
        tol: float = 1e-4
    ) -> None:

        valid_regularizations = {"None", "l1", "l2", "elastic_net"}
        if regularization not in valid_regularizations:
            raise ValueError(f"regularization must be one of {valid_regularizations}, got '{regularization}'")
        self.n_classes = n_classes
        self.n_features = n_features
        if random_state != 0: 
            np.random.seed(random_state)

        limit = np.sqrt(6 / (self.n_features + self.n_classes))
        self.theta = np.random.uniform(-limit, limit, size=(self.n_features + 1, self.n_classes))

        self.tol = tol
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
        self.l1_ratio = l1_ratio if regularization in {"l1","elastic_net"} else 0.0
        self.l2_ratio = l2_ratio if regularization in {"l2", "elastic_net"} else 0 

    def softmax(self, X: NDArray) -> NDArray:
        """
        Compute softmax probabilities across classes.

        Args:
            X: Feature matrix of shape (n_samples, n_features + 1).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        theta = self.theta.reshape(self.n_features + 1, self.n_classes)
        logits = X @ theta
        logits -= np.max(logits, axis=1, keepdims=True) 
        exp_scores = np.exp(logits)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)



    def cross_entropy(self, theta_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute the regularized cross-entropy loss.

        Args:
            theta_flat: Flattened parameter tensor (requires grad).

        Returns:
            Scalar torch tensor representing the loss.
        """
        X = torch.tensor(self.X_bias, dtype=torch.float64)
        Y = torch.tensor(self.Y, dtype=torch.long) 
        theta = theta_flat.view(self.n_features + 1, self.n_classes)

        logits = X @ theta 

        loss = F.cross_entropy(logits, Y)

        if self.regularization == "l2":
            loss += self.l2_ratio * torch.sum(theta[1:, :] ** 2)
        elif self.regularization == "l1":
            loss += self.l1_ratio * torch.sum(torch.abs(theta[1:, :]))
        elif self.regularization == "elastic_net":
            loss += self.l2_ratio * torch.sum(theta[1:, :] ** 2) + self.l1_ratio * torch.sum(torch.abs(theta[1:, :]))
        return loss
        
    def fit(self, X: NDArray[np.float64], Y: NDArray[np.float64]) -> "LogisticRegression":
        """
        Train the model using the selected solver.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            Y: Class label array of shape (n_samples,).

        Returns:
            Self (fitted model).
        """
        self.Y = Y 
        self.X = X
        self.X_bias = self._add_bias(X)

        if self.solver == "GD":
            self.theta = GradientDescent(self.cross_entropy, self.theta, self.lr, self.n_iter, self.tol)
        elif self.solver == "LBFGS":
            self.theta = LBFGS(self.cross_entropy, self.theta, self.lr, self.n_iter, self.m, self.tol)

        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        X = self._add_bias(X)
        return self.softmax(X)

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class labels.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted label array of shape (n_samples,).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _add_bias(self, X: NDArray) -> NDArray:
        """
        Append bias term (column of ones) to input matrix.

        Args:
            X: Input matrix of shape (n_samples, n_features).

        Returns:
            Augmented matrix of shape (n_samples, n_features + 1).
        """
        return np.hstack([np.ones((X.shape[0], 1)), X])
    

    def plot(self, i, j ,title='Logistic Regression Decision Boundary'):
        
        plt.scatter(self.X[:, i], self.X[:, j], c=self.Y, s=45, cmap='winter', alpha=0.9)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 500)
        yy = np.linspace(ylim[0], ylim[1], 500)

        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        if xy.shape[1] != self.X.shape[1]:
            padded_xy = np.zeros((xy.shape[0], self.X.shape[1]))
            padded_xy[:, [i, j]] = xy
            xy = padded_xy
        Z = self.predict(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, levels=[0, 1],linestyles=['-', '-'])
        plt.title(title)
        plt.show()

