import numpy as np
from numpy.typing import NDArray


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
    
    def confusion_matrix(self, X: NDArray, Y: NDArray):
        pred = self.predict(X)
        cm = np.zeros([self.n_classes, self.n_classes])

        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i == j:
                    cm[i,i] = np.sum((pred == np.unique(Y)[i]) & (Y == np.unique(Y)[i]))
                else:
                    cm[i,j] = np.sum((pred == np.unique(Y)[j]) & (Y == np.unique(Y)[i]))
        
        return cm
    
    def precision(self, X:NDArray, Y:NDArray):
        cm = self.confusion_matrix(X, Y)
        precision = []
        for i in range(self.n_classes):
            precision.append(
                cm[i,i]/np.sum(cm[:,i])
                )
        return np.array(precision)
    
    def recall(self, X:NDArray, Y:NDArray):
        cm = self.confusion_matrix(X, Y)
        recall = [] 
        for i in range(self.n_classes):
            recall.append(
                cm[i,i]/np.sum(cm[i,:])
                )
        return np.array(recall)
    def f1_stat(self, X: NDArray, Y: NDArray):
        f1 = 2/(1/self.precision(X, Y)+1/self.recall(X, Y))

        return f1