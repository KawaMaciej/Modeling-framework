from solvers.grad_methods import GradientDescent, LBFGS
import torch
import numpy as np
from numpy.typing import NDArray

class SVMClassificator:
    def __init__(self, n_iter: int=1000, lr: float=0.0001, C: float = 1.0) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.C = C

    def fit(self, X, Y):
        self.w = np.zeros(X.shape[1], dtype=np.float64) 
        X = torch.tensor(X, dtype=torch.float64)
        Y = torch.tensor(Y, dtype=torch.float64)
        w = torch.tensor(self.w, dtype=torch.float64, requires_grad=True)
        def _loss(w):
            margin = 1 - Y * (X @ w)
            hinge_loss = torch.max(torch.tensor(0), margin).mean()
            reg_loss = 0.5 * torch.dot(w, w)
            return reg_loss + self.C * hinge_loss
        
        self.w = LBFGS(_loss, w, self.lr, self.n_iter)
        
        return self
    
    def predict(self, X):
        return (X @ self.w> 0).astype(int)
