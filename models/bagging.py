import numpy as np
from scipy.stats import mode

class BaggingClassifier:
    def __init__(self, estimator, n_splits=4, soft_voting=False, replacement = True, random_state = 42 ) -> None:
        self.n_splits = n_splits
        self.estimator = estimator
        self.soft_voting= soft_voting
        self.replacement = replacement
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X, Y):
        self.models = []
        X = np.array(X)
        Y = np.array(Y)
        
        for _ in range(self.n_splits):
            indices = np.random.choice(len(X), size=len(X), replace=self.replacement)
            X_sample = X[indices]
            Y_sample = Y[indices]

            model = self.estimator
            model.fit(X_sample, Y_sample)
            self.models.append(model)

        return self
    
    def predict(self, X):
        if not self.soft_voting:
            predictions = np.array([model.predict(X) for model in self.models])
            majority_votes, _ = mode(predictions, axis=0, keepdims=False)
            return majority_votes

        else:
            probas = np.array([model.predict_proba(X) for model in self.models])
            avg_probas = np.mean(probas, axis=0)
            return np.argmax(avg_probas, axis=1)
        




class BaggingRegressor:
    def __init__(self, estimator, n_splits=4, replacement = True, random_state = 42, **model_args, ) -> None:
        self.n_splits = n_splits
        self.model_args = model_args  
        self.estimator = estimator
        self.replacement = replacement
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X, Y):
        self.models = []
        X = np.array(X)
        Y = np.array(Y)
        
        for _ in range(self.n_splits):
            indices = np.random.choice(len(X), size=len(X), replace=self.replacement)
            X_sample = X[indices]
            Y_sample = Y[indices]

            model = self.estimator(**self.model_args)
            model.fit(X_sample, Y_sample)
            self.models.append(model)

        return self
    
    def predict(self, X):

        probas = np.array([model.predict(X) for model in self.models])
        avg_probas = np.mean(probas, axis=0)
        return avg_probas