import numpy as np
from numpy.typing import NDArray
from models.logistic import LogisticRegression
from typing import Any
from copy import deepcopy

class OVA:
    def __init__(self, model_class: Any) -> None:
        #self.model_class = lambda: model_class
        self.model_class = model_class
        #model = deepcopy(self.model_class())
    def fit(self, X: NDArray, Y: NDArray) -> "OVA":
        self.models = []
        self.classes = np.unique(Y)
        for cls in self.classes:
            y_binary = np.where(Y == cls, 1, -1)
            model = deepcopy(self.model_class) 
            model.fit(X, y_binary)
            self.models.append(model)
        return self

    def predict(self, X: NDArray) -> NDArray:
        scores = np.array([model.predict(X) for model in self.models])
        return self.classes[np.argmax(scores, axis=0)]
