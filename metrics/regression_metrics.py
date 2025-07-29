import numpy as np
from numpy.typing import NDArray



def MSE(Y: NDArray, pred: NDArray) -> float:

    mse = float(np.mean((Y-pred)**2))

    return mse
    
def RMSE(Y: NDArray, pred: NDArray) -> float:

    rmse = np.sqrt(
            MSE(pred, Y)
        )
        
    return float(rmse)

    
def MAE(Y: NDArray, pred: NDArray) -> float:

    mae = float(
            np.mean(
                np.abs(
            pred - Y
        )))

    return mae

def R2_score(Y: NDArray, pred: NDArray ) -> float:

    y_mean = Y.mean()
    return float(1 - np.sum((Y - pred) ** 2) / np.sum((Y - y_mean) ** 2))

def R_adjusted(X:NDArray, Y:NDArray, pred:NDArray) -> float:
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    r_adj = 1 - (1-R2_score(Y, pred)*(X.shape[0]-1)/(X.shape[0]+X.shape[1]-1))
    return float(r_adj)