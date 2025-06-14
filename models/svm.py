
import jax.numpy as jnp
from jax import jit, grad
import numpy as np

class SVMClassificator:
    def __init__(self, n_iter: int=1000, lr: float=0.0001, C: float = 1.0) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.C = C

    def fit(self, X, Y):
        def _loss(w):
            margin = 1 - Y * (X @ w)
            hinge_loss = jnp.maximum(0, margin).mean()
            reg_loss = 0.5 * jnp.dot(w, w)
            return reg_loss + self.C * hinge_loss
        
        loss_grad = jit(grad(_loss))
        self.w = jnp.zeros(X.shape[1])
        
        for _ in range(self.n_iter):
            grads = loss_grad(self.w)
            self.w -= self.lr * grads
        if np.any(np.isnan(self.w)):
            raise ValueError("Gradient explosion, please change learning rate")
        
        return self
    
    def predict(self, X):
        return (X @ self.w> 0).astype(int)
