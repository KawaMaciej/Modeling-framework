import jax.numpy as jnp
from jax import grad, jit
from typing import Tuple

class LassoRegression:
    def __init__(self, alpha: float = 1.0, lr: float = 0.01, n_iter: int = 1000):
        """
        Initialize the Lasso Regression model.

        Parameters:
        ----------
        alpha : float
            Regularization strength (L1 penalty term).
        lr : float
            Learning rate for gradient descent.
        n_iter : int
            Number of iterations.
        """
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.weights = jnp.ndarray

    def _loss(self, weights: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Lasso loss: MSE + L1 penalty
        """
        y_pred = X @ weights
        mse = jnp.mean((y - y_pred) ** 2)
        l1 = self.alpha * jnp.sum(jnp.abs(weights[1:])) 
        return mse + l1

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> "LassoRegression":
        """
        Fit the model using gradient descent with JAX autograd.

        Parameters:
        ----------
        X : jnp.ndarray
            Feature matrix (n_samples, n_features).
        y : jnp.ndarray
            Target vector (n_samples,).
        """
        X = jnp.c_[jnp.ones((X.shape[0], 1)), X] 
        self.weights = jnp.zeros(X.shape[1])

        loss_grad = jit(grad(self._loss))

        for _ in range(self.n_iter):
            grads = loss_grad(self.weights, X, y)
            self.weights -= self.lr * grads

        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict target values.

        Parameters:
        ----------
        X : jnp.ndarray
            Feature matrix (n_samples, n_features).
        """
        X = jnp.c_[jnp.ones((X.shape[0], 1)), X]
        return X @ self.weights
    @property
    def coef(self) -> jnp.ndarray:
        return self.weights[1:]
    @property
    def intercept(self) -> float:
        return self.weights[0]
    
