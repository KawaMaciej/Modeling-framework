
import numpy as np
from numpy.typing import NDArray

class OVO:
    def __init__(self, model_class, **model_args) -> None:
        self.model_class = model_class
        self.model_args = model_args  

    def fit(self, X: NDArray, Y: NDArray):
        self.classes_ = np.unique(Y)
        self.models = []

        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                class_i = self.classes_[i]
                class_j = self.classes_[j]

                mask = np.logical_or(Y == class_i, Y == class_j)
                X_pair = X[mask]
                Y_pair = Y[mask]

                Y_binary = (Y_pair == class_j).astype(int)

                model = self.model_class(**self.model_args)  
                model.fit(X_pair, Y_binary)

                self.models.append((model, (class_i, class_j)))

        return self
    
    def predict(self, X: NDArray) -> NDArray:
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes), dtype=int)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        for model, (class_i, class_j) in self.models:
            preds = model.predict(X)
            idx_i = class_to_idx[class_i]
            idx_j = class_to_idx[class_j]
            votes[np.arange(n_samples), np.where(preds == 0, idx_i, idx_j)] += 1
        
        return self.classes_[np.argmax(votes, axis=1)]