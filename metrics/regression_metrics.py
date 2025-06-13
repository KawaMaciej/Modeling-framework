import numpy as np
from numpy.typing import NDArray



def MSE(Y: NDArray, pred: NDArray) -> float:

    mse = float(np.mean((Y-pred)**2))

    return mse
    
def RMSE(Y: NDArray, pred: NDArray) -> float:

    rmse = np.sqrt(
            MSE(pred, Y)
        )
        
    return rmse
    
def MAE(Y: NDArray, pred: NDArray) -> float:

    mae = float(
            np.mean(
                np.abs(
            pred - Y
        )))

    return mae