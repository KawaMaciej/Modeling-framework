
import numpy as np
from numpy.typing import NDArray

class OVO:
    def __init__(self, model_class) -> None:
        self.model_class = lambda: model_class

    def fit(self, X: NDArray, Y: NDArray):
        self.classes_ = np.unique(Y)
        self.models = []
        self.pairs = []

        for i in range(len(self.classes_)):
            for j in range(i + 1, len(self.classes_)):
                class_i = self.classes_[i]
                class_j = self.classes_[j]
                self.pairs.append((class_i, class_j))

                mask = np.logical_or(Y == class_i, Y == class_j)
                X_pair = X[mask]
                Y_pair = Y[mask]

                Y_binary = (Y_pair == class_j).astype(int)

                model = self.model_class()   
                model.fit(X_pair, Y_binary)

                self.models.append((model, (class_i, class_j)))

        return self

    def predict(self, X: NDArray) -> NDArray:
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes_)), dtype=int)

        for model, (i, j) in self.models:
            preds = model.predict(X)
            for idx, pred in enumerate(preds):
                voted_class = j if pred == 1 else i
                class_idx = np.where(self.classes_ == voted_class)[0][0]
                votes[idx, class_idx] += 1

        return self.classes_[np.argmax(votes, axis=1)]