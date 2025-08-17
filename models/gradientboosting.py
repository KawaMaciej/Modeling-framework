import numpy as np
import copy

class GradientBoostingRegressor:
    def __init__(self, base_model, n_estimators=3, lr=0.1, verbose=False):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = lr
        self.verbose = verbose

    def fit(self, X, y):
        self.models = []
        y = np.array(y, copy=True)

        self.init_pred = np.mean(y)
        residual = y - self.init_pred

        for i in range(self.n_estimators):
            model = copy.deepcopy(self.base_model)  # fresh copy
            model.fit(X, residual)
            self.models.append(model)

            pred = model.predict(X)
            residual -= self.learning_rate * pred

            if self.verbose:
                mse = np.mean(residual**2)
                print(f"Iteration {i+1}/{self.n_estimators}, Residual MSE: {mse:.4f}")

        return self

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_pred)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
    


class GradientBoostingClassifier:
    def __init__(self, base_model, n_estimators=3, lr=0.1, verbose=False):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = lr
        self.verbose = verbose

    def fit(self, X, y):
        self.models = []
        y = np.array(y, copy=True, dtype=int)

        p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.init_pred = np.log(p / (1 - p))

        scores = np.full(y.shape, self.init_pred)

        for i in range(self.n_estimators):
            p = 1 / (1 + np.exp(-scores))
            residual = y - p 
            model = copy.deepcopy(self.base_model)
            model.fit(X, residual)
            self.models.append(model)

            scores += self.learning_rate * model.predict(X)

            if self.verbose:
                loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                print(f"Iteration {i+1}/{self.n_estimators} | Log-loss: {loss:.4f}")

            return self

    def predict_proba(self, X):
        scores = np.full(X.shape[0], self.init_pred)
        for model in self.models:
            scores += self.learning_rate * model.predict(X)
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)