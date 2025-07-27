from typing import List
from scipy.stats import mode
import numpy as np

class VotingClassifier:
    def __init__(self, estimators: List, soft_votting = False) -> None:

        self.estimators = estimators
        self.soft_votting = soft_votting

    def voters(self):
        
        return self.estimators
    
    def fit(self, X, y):
        self.predictors = []
        for estimator in self.estimators:
            estimator.fit(X,y)
            self.predictors.append(estimator)

        return self
    
    def predict(self, X):
        if not self.soft_votting:
            predictions = np.array([model.predict(X) for model in self.predictors])
            majority_votes, _ = mode(predictions, axis=0, keepdims=False)
            return majority_votes

        else:
            probas = np.array([model.predict_proba(X) for model in self.predictors])
            avg_probas = np.mean(probas, axis=0)
            return np.argmax(avg_probas, axis=1)