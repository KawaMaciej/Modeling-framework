# Linear Regression & Logistic Regression Models

This repository contains custom implementations of Linear Regression and Logistic Regression (including regularization), as well as tools for model diagnostics and classification metrics. The code is organized for educational and experimental purposes, with a focus on understanding the math and workflow behind regression and classification.

## Features

- **Linear Regression**: Supports OLS, Ridge, Lasso, and ElasticNet regularization.
- **Logistic Regression**: Multiclass (softmax) with optional L1/L2 regularization.
- **Diagnostics**: Residual plots, Ramsey RESET test, VIF for multicollinearity, Cook's distance, and more.
- **Classification Metrics**: Confusion matrix, precision, recall, F1, accuracy, balanced accuracy, NPV, FOR, Fowlkes-Mallows.
- **One-vs-All (OVA)**: For multiclass classification.
- **Jupyter Notebook**: Example usage and testing.

## Directory Structure

```
repo/
│
├── models/
│   ├── linear.py           # LinearRegression class
│   ├── logistic.py         # LogisticRegression class
│   └── onevsall.py         # OVA multiclass wrapper
│
├── metrics/
│   └── classification_metrics.py  # Classification metric functions
│
├── tests/
│   ├── RESET_test.py       # Ramsey RESET test
│   ├── Breusch_Pagan.py    # Breusch-Pagan test
│   └── DurbinWatson_test.py# Durbin-Watson test
│
├── linear_regression_testing.ipynb # Main notebook for experiments
└── README.md
```

## Getting Started

1. **Clone the repository**
    ```bash
    git clone https://github.com/KawaMaciej/Modeling-framework.git
    cd repo
    ```

2. **Install dependencies**
    - Python 3.8+
    - numpy, pandas, plotly, jax, scikit-learn (for some utilities)
    - (Optional) Jupyter Notebook

    You can install requirements with:
    ```bash
    pip install numpy pandas plotly jax scikit-learn
    ```

3. **Run the notebook**
    Open `linear_regression_testing.ipynb` in Jupyter Notebook or VS Code.

## Example Usage

```python
from models.linear import LinearRegression
import numpy as np

X = np.random.normal(0, 1, (100, 3))
Y = 2*X[:,0] - X[:,1] + np.random.normal(0, 0.5, 100)

reg = LinearRegression(regularization="Ridge", alpha=0.1)
reg.fit(X, Y)
print("Coefficients:", reg.coef)
print("Intercept:", reg.intercept)
```

## Notes

- All models are implemented from scratch for learning purposes.
- Diagnostics and metrics are also custom implementations.
- See the notebook for detailed examples and tests.

