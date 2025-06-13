import numpy as np
from numpy.typing import NDArray
from models.logistic import LogisticRegression

class OVA:
    def __init__(self,
        n_iter: int = 1000,
        lr: float = 0.001,
        regularization: str = "None",
        alpha: float = 0.2
    ) -> None:

        valid_regularizations = {"None", "l1", "l2"}
        if regularization not in valid_regularizations:
            raise ValueError(f"regularization must be one of {valid_regularizations}, got '{regularization}'")
        
        self.n_iter = n_iter
        self.lr = lr

        self.regularization = regularization
        self.alpha = alpha if regularization in {"l1", "l2"} else 0.0

    def fit(self, X: NDArray, Y: NDArray) -> "OVA":
        data = []
        for i in np.unique(Y):
            data.append((Y == i).astype(int))
        self.models = []
        for i in data:
            self.models.append(LogisticRegression(X.shape[1], 
                                             2, 
                                             regularization=self.regularization,
                                             n_iter=self.n_iter,
                                             lr = self.lr,
                                             alpha = self.alpha
                                             ).fit(X, i))
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        preds = []
        for model in self.models:
            preds.append(model.predict_proba(X))
        preds = np.array(preds)
        return preds[:,:,1].argmax(axis=0)
