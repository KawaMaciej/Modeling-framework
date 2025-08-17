# 🧠 Modeling Framework

A modular and extensible Python framework for building and experimenting with machine learning models — including custom implementations of linear regression, Lasso, Ridge, and ElasticNet using gradient descent and many others.


---

## 📦 Features

### Regression Models
- **Linear Regression** (`linear.py`)
  - Ordinary Least Squares (OLS)
  - Ridge regularization
  - Lasso regularization
  - ElasticNet regularization
  - Custom gradient descent optimizer 
  - Model diagnostics: residual plots, linearity (RESET), homoscedasticity (Breusch-Pagan), independence (Durbin-Watson), multicollinearity (VIF), normality (Shapiro-Wilk)
  - Error metrics: MAE, RMSE, MSE, R², Adjusted R²
  - Cross-validation (KFold)
  - Outlier detection (Cook's distance)
  - Interactive visualization with Plotly

### Classification Models
- **Logistic Regression** (`logistic.py`)
  - Multiclass support
  - L1, L2, and ElasticNet regularization
  - Gradient Descent and LBFGS optimizers
  - Decision boundary plotting

- **Support Vector Machine (SVM)** (`svm.py`)
  - Linear, polynomial, sigmoid, and RBF kernels
  - Custom gradient descent optimizer
  - Decision function and probability prediction
  - Decision boundary plotting

- **Decision Trees** (`decision_trees.py`)
  - Classification and regression trees
  - Gini impurity and MSE splitting
  - Probability prediction for classification

- **Bagging** (`bagging.py`)
  - BaggingClassifier and BaggingRegressor
  - Hard and soft voting
  - Bootstrap sampling

- **Voting Classifier** (`votingclassifier.py`)
  - Hard and soft voting across multiple estimators

- **One-vs-All (OVA)** (`onevsall.py`)
  - Multiclass strategy for any binary classifier

- **One-vs-One (OVO)** (`onevsone.py`)
  - Multiclass strategy for any binary classifier

### Preprocessing & Utilities
- **Preprocessing** (`preprocessing/data_mods.py`)
  - Train/test split
  - Categorical encoding
  - Feature scaling (standardization)

- **Metrics** (`metrics/regression_metrics.py`, `metrics/classification_metrics.py`)
  - Regression and classification metrics

- **Solvers** (`solvers/grad_methods.py`)
  - Custom gradient descent and LBFGS implementations

- **Statistical Tests** (`tests/`)
  - Ramsey RESET, Breusch-Pagan, Durbin-Watson

---



## Notes

- All models are implemented from scratch for learning purposes.
- Diagnostics and metrics are also custom implementations.
- See the notebook for detailed examples and tests.
