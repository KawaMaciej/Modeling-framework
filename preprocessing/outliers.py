import numpy as np
from numpy.typing import NDArray
from metrics.regression_metrics import MSE

def Cooks_distance(model, X: NDArray, Y: NDArray) -> NDArray:
        D = []
        pred = model.predict(X)
        for i in range(X.shape[0]):
            X_without_ith = np.delete(X, i, axis = 0)
            Y_without_ith = np.delete(Y, i, axis = 0)
            pred_without_ith = np.delete(pred, i, axis = 0)
            model = model.fit(X_without_ith, Y_without_ith)
            square = (pred_without_ith - model.predict(X_without_ith))**2
            D.append(np.sum(square)/(MSE(Y, pred)*X.shape[1]))
        return np.array(D)