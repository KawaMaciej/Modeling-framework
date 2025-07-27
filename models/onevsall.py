import numpy as np
from numpy.typing import NDArray
from models.logistic import LogisticRegression
from typing import Any
from copy import deepcopy

class OVA:
    def __init__(self, model_instance: Any) -> None:
        self.model_instance = model_instance

    def fit(self, X: NDArray, Y: NDArray) -> "OVA":
        self.models = []
        self.classes = np.unique(Y)
        for cls in self.classes:
            model = deepcopy(self.model_instance)
            if hasattr(model, "decision_function"):
                y_binary = np.where(Y == cls, 1, -1) 
                self.use_neg1_pos1 = True
            elif hasattr(model, "cross_entropy"):
                y_binary = np.where(Y == cls, 1, 0) 
                self.use_neg1_pos1 = False
            elif hasattr(model, "_best_split"):
                y_binary = np.where(Y == cls, 1, 0) 
                self.use_neg1_pos1 = False
            else:
                raise ValueError("Model must support either `decision_function`, `predict_proba` or `_best_split`")
            
            model.fit(X, y_binary)
            self.models.append(model)
        return self

    def predict(self, X: NDArray) -> NDArray:
        if hasattr(self.model_instance, "decision_function"):
            scores = np.array([model.decision_function(X) for model in self.models])
        elif hasattr(self.model_instance, "predict_proba"):
            scores = np.array([model.predict(X) for model in self.models]) 
        elif hasattr(self.model_instance, "_best_split"):
            scores = np.array([model.predict(X) for model in self.models]) 
        else:
            raise ValueError("Model must support either `decision_function` or `predict_proba`")
        
        return self.classes[np.argmax(scores, axis=0)]
