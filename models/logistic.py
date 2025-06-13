import numpy as np
from numpy.typing import NDArray
from metrics.classification_metrics import *

class LogisticRegression:
    def __init__(self, n_features: int, n_classes: int, n_iter: int=1000, lr: float = 0.001) -> None:
        self.theta: NDArray = np.zeros((n_features, n_classes)) 
        self.n_iter = n_iter
        self.lr = lr
        self.n_classes = n_classes

    def score_function(self, x: NDArray, k: int):
        return np.dot(x, self.theta[:, k])
    
    def softmax(self, X: NDArray):
        logits = X @ self.theta
        logits -= np.max(logits, axis=1, keepdims=True)  
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs


    def cross_entropy(self, X: NDArray):
        pass

    def fit(self, X: NDArray, Y: NDArray):
        Y = np.eye(self.n_classes)[Y]

        for _ in range(self.n_iter):
            probs = self.softmax(X)
            error = probs - Y
            grad = X.T @ error / X.shape[0]
            self.theta -= self.lr * grad

        return self

    def predict(self, X: NDArray) -> NDArray:
        preds = self.softmax(X)
        return np.argmax(preds, axis=1)
    
