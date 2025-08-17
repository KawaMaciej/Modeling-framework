import numpy as np
from numpy.typing import NDArray
from metrics.classification_metrics import *
import torch
from solvers.grad_methods import *
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
        solver (str): Optimization algorithm, either 'GD', 'LBFGS', 'ADABelief', 'Lion' .
        m (int): History size for LBFGS.
        random_state (int): Seed for reproducibility.
        tol (float): Tolerance for convergence (Δ loss threshold).

    Attributes:
        theta (NDArray[np.float64]): Weight matrix of shape (n_features + 1, n_classes).
    """

    def __init__(
        self,
        n_iter: int = 1000,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        regularization: str = "None",
        l1_ratio: float = 0.5,
        l2_ratio: float = 0.5,
        solver: str = "GD",
        m: int = 10,
        random_state: int = 0,
        tol: float = 1e-4, 
        verbose = True
    ) -> None:

        valid_regularizations = {"None", "l1", "l2", "elastic_net"}
        if regularization not in valid_regularizations:
            raise ValueError(f"regularization must be one of {valid_regularizations}, got '{regularization}'")
        

        if random_state != 0: 
            np.random.seed(random_state)



        self.verbose = verbose
        self.solver = solver
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.m = m
        self.weight_decay = weight_decay

        self.regularization = regularization
        self.l1_ratio = l1_ratio if regularization in {"l1","elastic_net"} else 0.0
        self.l2_ratio = l2_ratio if regularization in {"l2", "elastic_net"} else 0 
        
        self.optimizers = {
            "GD": GradientDescent,
            "LBFGS": LBFGS,
            "ADABelief": AdaBeliefOptimizer,
            "Lion": LionOptimizer
            }
    def __call__(self):

        return self
    
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
        Compute the regularized cross-entropy loss with optional sample weighting.

        Args:
            theta_flat: Flattened parameter tensor.

        Returns:
            Weighted, regularized cross-entropy loss (scalar tensor).
        """


        X = torch.tensor(self.X_bias, dtype=torch.float64)
        Y = torch.tensor(self.Y, dtype=torch.long)
        weights = torch.tensor(self.sample_weight, dtype=torch.float64) if hasattr(self, 'sample_weight') else torch.ones_like(Y, dtype=torch.float64)

        theta = theta_flat.view(self.n_features + 1, self.n_classes)
        logits = X @ theta  
        log_probs = F.log_softmax(logits, dim=1)

        log_likelihood = log_probs[torch.arange(X.shape[0]), Y]
        weighted_loss = -weights * log_likelihood
        loss = weighted_loss.sum() / weights.sum()

        if self.regularization == "l2":
            loss += self.l2_ratio * torch.sum(theta[1:, :] ** 2)
        elif self.regularization == "l1":
            loss += self.l1_ratio * torch.sum(torch.abs(theta[1:, :]))
        elif self.regularization == "elastic_net":
            loss += self.l2_ratio * torch.sum(theta[1:, :] ** 2) + self.l1_ratio * torch.sum(torch.abs(theta[1:, :]))

        return loss
        
    def fit(self, X: NDArray[np.float64], Y: NDArray[np.float64], sample_weight=None) -> "LogisticRegression":
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(Y)) if Y.ndim == 1 else Y.shape[1]

        limit = np.sqrt(6 / (self.n_features + self.n_classes))
        self.theta = np.random.uniform(-limit, limit, size=(self.n_features + 1, self.n_classes))

        self.Y = Y
        self.X = X
        self.sample_weight = sample_weight if sample_weight is not None else np.ones_like(Y, dtype=np.float64)
        self.X_bias = self._add_bias(X)

        loss_fn = self.cross_entropy
        opt_func = self.optimizers.get(self.solver)

        if opt_func is None:
            raise ValueError(f"Unknown optimization method: {self.solver}")

        self.theta = opt_func(
            loss_fn,
            init_x=self.theta,
            lr=self.lr,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=self.verbose,
            **({"weight_decay": self.weight_decay} if self.solver == "ADABelief" else {})
        )

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

