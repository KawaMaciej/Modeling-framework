import numpy as np


class AdaBoostClassifier:
    def __init__(self,
                 model_class,
                 n_estimators,
                 verbose=True
                 ) -> None:
        self.model_class = model_class
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.verbose = verbose
    def fit(self, X, Y):
        Y = 2 * Y - 1 
        w = np.ones(X.shape[0]) / X.shape[0]

        for i in range(self.n_estimators):
            model = self.model_class()
            model.fit(X, ((Y + 1) // 2))
            pred = model.predict(X)
            pred = 2 * pred - 1 

            err = np.sum(w * (pred != Y)) / np.sum(w)

            if err == 0:
                alpha = 10  
            elif err >= 0.5:
                if self.verbose == True:
                    print(f"⚠️ Estimator {i}: weak learner has error {err:.4f} (≥ 0.5). Skipping.")
                continue
            else:
                alpha = 0.5 * np.log((1 - err) / err)

            self.models.append(model)
            self.alphas.append(alpha)

            w *= np.exp(-alpha * Y * pred)
            w /= np.sum(w)  

    def predict(self, X):
        pred = sum(alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.models))
        return np.sign(pred)
    
    def predict_proba(self, X):
        pred = sum(alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.models))
        return pred