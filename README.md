# Linear Regression from Scratch 🧮

A clean and extensible implementation of **Linear Regression** in Python, supporting:
- Ordinary Least Squares (Normal Equation)
- Ridge (L2) regularization
- Lasso (L1) regression via gradient descent using JAX

---

## 🚀 Features

- **Fit** a linear model with intercept using closed-form solution or regularized optimization
- **Predict** new data points
- **Score** model performance (R²)
- **Diagnostics**: residuals, errors (MAE, RMSE, MSE)
- **Residual plot** via Plotly for visual inspection
- **Model assumption checks**: linearity, homoscedasticity, autocorrelation, multicollinearity, and residual normality
- Easy to **extend**—add new methods like confidence intervals, CV, etc.

---

## 🔧 Requirements

- Python 3.9+
- Core libraries:
  - `numpy`, `jax`, `scipy`, `plotly`, `tabulate`
- Test-dependents (optional):
  - `statsmodels`, `sklearn`
- Install with:
  ```bash
  pip install numpy jax scipy plotly tabulate
