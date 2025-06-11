import numpy as np
from numpy.typing import NDArray
import plotly.graph_objs as go
from tests.RESET_test import RESET_test
from tests.Breusch_Pagan import Breusch_Pagan
from tests.DurbinWatson_test import DurbinWatson
from scipy import stats
from tabulate import tabulate
import jax.numpy as jnp
from jax import grad, jit
import pandas as pd


class LinearRegression:
    """
    A simple implementation of Linear Regression supporting normal equation, Ridge, and Lasso regularization.
    Includes diagnostics like residual plots, assumption checks, and error metrics.
    """

    def __init__(self, regularization: str = "None", alpha: float = 0.1, n_iter: int = 1000, lr: float = 0.0001) -> None:
        """
        Initialize the LinearRegression model.

        Parameters:
        -----------
        regularization : str
            Type of regularization ('None', 'Ridge', or 'Lasso').
        alpha : float
            Regularization strength (used for Ridge and Lasso).
        n_iter : int
            Number of iterations for Lasso gradient descent.
        lr : float
            Learning rate for Lasso gradient descent.
        """
        self.regularization = regularization
        self.beta: NDArray | None = None
        self.feature_names: list | None = None
        if self.regularization  == "Lasso":
            self.alpha = alpha
            self.n_iter = n_iter
            self.lr = lr
        if self.regularization  == "Ridge":
            self.alpha = alpha

    def __repr__(self) -> str:
        """
        Return a string representation of the model.
        """
        return f"{self.__class__.__name__}{self.regularization}(fit_intercept=True)"

    def fit(self, X: NDArray, Y: NDArray ) -> "LinearRegression":
        """
        Fit the linear regression model based on selected regularization.

        Parameters:
        -----------
        X : NDArray
            Feature matrix of shape (n_samples,) or (n_samples, n_features).
        Y : NDArray
            Target vector of shape (n_samples,) or (n_samples, 1).

        Returns:
        --------
        LinearRegression
            The fitted model instance.
        """
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'x{i}' for i in range(X.shape[1])]
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        is_numeric = np.issubdtype(self.X.dtype, np.number)
        if is_numeric == False:
            raise ValueError("Data is not numerical.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x = np.c_[np.ones((self.X.shape[0], 1)), self.X] 
        def _loss(weights: jnp.ndarray, arg: jnp.ndarray, y: jnp.ndarray) -> float:
            """
            Lasso loss: MSE + L1 penalty
            """
            y_pred = arg @ weights
            mse = jnp.mean((y - y_pred) ** 2)
            l1 = self.alpha * jnp.sum(jnp.abs(weights[1:])) 
            return mse + l1

        if self.regularization =="None":
            self.beta = np.linalg.pinv(x.T @ x) @ x.T @ Y  
        if self.regularization =="Ridge":
            self.beta = np.linalg.pinv(x.T @ x + self.alpha * np.ones_like(x.T @ x) ) @ x.T @ Y
        if self.regularization =="Lasso":
            loss_grad = jit(grad(_loss))
            self.beta = jnp.zeros(x.shape[1])

            for _ in range(self.n_iter):
                grads = loss_grad(self.beta, x, self.Y)
                self.beta -= self.lr * grads

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict target values using the fitted model.

        Parameters:
        -----------
        X : NDArray
            Feature matrix.

        Returns:
        --------
        NDArray
            Predicted values.
        """

        if self.beta is None:
            raise ValueError("Model is not fitted yet.")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] + 1 == self.beta.shape[0]: 
            X = np.c_[np.ones((X.shape[0], 1)), X]
        return X @ self.beta

    def score(self, X: NDArray, Y: NDArray) -> float:
        """
        Compute the R^2 score (coefficient of determination).

        Parameters:
        -----------
        X : NDArray
            Feature matrix.
        Y : NDArray
            True target values.

        Returns:
        --------
        float
            R^2 score.
        """
        y_mean = Y.mean()
        y_predicted = self.predict(X)
        return 1 - np.sum((Y - y_predicted) ** 2) / np.sum((Y - y_mean) ** 2)

    def resid(self, X: NDArray, Y: NDArray) -> NDArray:
        """
        Compute model residuals.

        Parameters:
        -----------
        X : NDArray
            Feature matrix.
        Y : NDArray
            True target values.

        Returns:
        --------
        NDArray
            Residuals.
        """
        return np.asarray(Y - self.predict(X))

    def plot(self, X: NDArray, Y:NDArray) -> None:
        """
        Plot features against target with fitted regression line using Plotly.

        Parameters:
        -----------
        X : NDArray
            Feature matrix.
        Y : NDArray
            Target values.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim > 1:
            Y = Y.flatten()
        n_features = X.shape[1]

        for i in range(n_features):
            Xi = X[:, i].reshape(-1, 1)
            Xi_line = np.linspace(Xi.min(), Xi.max(), 100).reshape(-1, 1)

            model_temp = LinearRegression()
            model_temp.fit(Xi, Y)
            Yi_line = model_temp.predict(Xi_line)

            scatter = go.Scatter(x=Xi.flatten(), y=Y, mode='markers', name='Data Points')
            line = go.Scatter(x=Xi_line.flatten(), y=Yi_line, mode='lines', name='Regression Line')

            fig = go.Figure(data=[scatter, line])
            fig.update_layout(
                title=f'Feature {i + 1} vs Y',
                xaxis_title=f'X{i + 1}',
                yaxis_title='Y'
            )
            fig.show()
            
    def plot_residuals(self, X: NDArray, Y: NDArray) -> None:
        """
        Create a residuals vs fitted values plot using Plotly.

        Parameters:
        -----------
        X : NDArray
            Feature matrix.
        Y : NDArray
            Target values.
        """
        y_pred = self.predict(X)
        residuals = self.resid(X, Y)

        scatter = go.Scatter(
                    x=y_pred.flatten(),
                    y=residuals.flatten(),
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='rgba(99, 110, 250, 0.7)', line=dict(width=1, color='DarkSlateGrey'))
                )

        zero_line = go.Scatter(
                    x=[y_pred.min(), y_pred.max()],
                    y=[0, 0],
                    mode='lines',
                    name='Zero Residual Line',
                    line=dict(color='red', dash='dash')
                )

        layout = go.Layout(
                    title='Residuals vs Fitted Values',
                    xaxis=dict(title='Fitted Values'),
                    yaxis=dict(title='Residuals'),
                    showlegend=True
                )

        fig = go.Figure(data=[scatter, zero_line], layout=layout)
        fig.show()

    @property
    def coef(self) -> NDArray:
        """
        Get model coefficients (excluding intercept).

        Returns:
        -------
        NDArray
            Coefficients.
        """
        if self.beta is None:
            raise ValueError("Model is not fitted yet.")
        return self.beta[1:]

    @property
    def intercept(self) -> NDArray:
        """
        Get model intercept.

        Returns:
        -------
        NDArray
            Intercept term.
        """
        if self.beta is None:
            raise ValueError("Model is not fitted yet.")
        return self.beta[:1]

    @property
    def Beta(self) -> NDArray:
        """
        Get the full beta vector (intercept + coefficients).

        Returns:
        -------
        NDArray
            Full parameter vector.
        """
        if self.beta is None or self.regularization == "Lasso":
            raise ValueError("Model is not fitted yet.")
        return self.beta

    def check_Linearity(self, X: NDArray, Y: NDArray, power: int = 2, make_plots: bool = False) -> None:
        """
        Perform the Ramsey RESET test for linearity.

        Parameters:
        ----------
        X : NDArrays
            Feature matrix.
        Y : NDArray
            Target values.
        power : int
            Maximum power for augmentation (default is 2).

        Returns:
        -------
        Print messege stating test result
        """
        if make_plots:
            self.plot(X, Y)
        F_stat, P_value = RESET_test(X, Y, self, power).run()
        if P_value > 0.05:
            mess = "The model is likely correctly specified ✅"
        else:
            mess = "Model may be misspecified (nonlinearity exists) ❌ "

        print(f"""================================================================================
        The Ramsey RESET test for linearity
        Test Statistic : {round(float(F_stat),4)} 
        P-value        : {round(float(P_value),4)} 
        Interpretation : {mess}
================================================================================
        """)
    
    def check_homoscedasticity(self, X: NDArray, Y: NDArray) -> None:
        """
        Check for homoscedasticity using the Breusch-Pagan test.

        Parameters:
        -----------
        X : NDArray
             Feature matrix.
        Y : NDArray
            Target values.

        Prints:
        -------
        Breusch-Pagan test statistic, p-value, and interpretation.
        """
        bp_stat, p_value = Breusch_Pagan(X, Y, self).run()

        if p_value < 0.05:
            message = "Heteroscedasticity detected — variance of residuals is not constant. ❌"
        else:
            message = "No evidence of heteroscedasticity — residuals appear homoscedastic. ✅"

        print(f"""================================================================================
        Breusch-Pagan Test for Heteroscedasticity
        Test Statistic : {round(float(bp_stat), 4)}
        P-value        : {round(float(p_value), 4)}
        Interpretation : {message}
================================================================================
        """)

    def check_independence(self, X: NDArray, Y: NDArray) -> None:
        """
        Check for autocorrelation in residuals using the Durbin-Watson test.

        This method evaluates whether residuals from the regression are independent,
        which is a key assumption in linear regression. It computes the Durbin-Watson
        statistic and prints an interpretation based on its value:

        - DW ≈ 2: No autocorrelation (ideal case)
        - DW < 1.5: Possible positive autocorrelation
        - DW > 2.5: Possible negative autocorrelation

        Parameters:
        ----------
        X : NDArray
            Feature matrix of shape (n_samples, n_features).
        Y : NDArray
            True target values of shape (n_samples,).

        Prints:
        -------
        Durbin-Watson statistic and interpretation of autocorrelation.
        """

        d = DurbinWatson(X, Y, self).run()

        if d is None:
            interpretation = "Durbin-Watson statistic could not be computed. ❌"
        elif 1.5 < d < 2.5:
            interpretation = "No autocorrelation ✅"
        elif d < 1.5:
            interpretation = "Possible positive autocorrelation ❌"
        else:
            interpretation = "Possible negative autocorrelation ❌"

        print(f"""================================================================================
        Durbin-Watson Test for Independence of Errors
        DW Statistic  : {round(d, 4) if d is not None else 'None'}
        Interpretation: {interpretation} 
    ================================================================================
        """)
        
    def check_multicollinearity(self, X: NDArray, Y: NDArray) -> None:
        """
        Check for multicollinearity using Variance Inflation Factor (VIF).
        Parameters:
        -----------
        X : NDArray
            Feature matrix.
        Y : NDArray
            Target values (not used for VIF calculation).
        Prints:
        -------
        VIF values and interpretation for each feature.
        """
        vifs = []
        interpretation = []
        n_features = X.shape[1]
        X = np.asarray(X)
        Y = np.asarray(Y)
        for i in range(n_features):
            X_others = np.delete(X, i, axis=1)
            y_target = X[:, i] 
            
            if X_others.shape[1] == 0:
                vif = 1.0
            else:
                model = LinearRegression().fit(X_others, y_target)
                r2 = model.score(X_others, y_target)
                vif = 1 / (1 - r2) if r2 < 1 else np.inf
            
            vifs.append(vif)
            
            if vif > 5:
                interpretation.append("Multicollinearity ❌")
            else:
                interpretation.append("No Multicollinearity ✅")
        
        print("================================================================================")
        print("        VIF Test for Multicollinearity")
        for idx, (vif, interp) in enumerate(zip(vifs, interpretation)):
            print(f"        Feature {idx+1}: VIF = {vif} | {interp}")
        print("================================================================================")
    
    def check_normality_of_resid(self, X: NDArray, Y: NDArray) -> None:
        resid = self.resid(X, Y)
        shapiro_result = float(stats.shapiro(resid).pvalue)

        if shapiro_result > 0.05:
            interpretation = "Residuals are from normal distribution ✅"
        else:
            interpretation = "Distribution of residuals is not normal ❌"
        print("================================================================================")
        print("        Shapiro-Wilk normality test of residuals \n")
        print(f"        {interpretation}")
        print("================================================================================")

    def run_assumptions(self, X: NDArray, Y: NDArray) -> None:
        if self.beta is not None:
            self.check_Linearity(X, Y, 2)
            self.check_homoscedasticity(X, Y)
            self.check_independence(X,Y)
            self.check_multicollinearity(X, Y)
            self.check_normality_of_resid(X, Y)
    
    def MSE(self, X: NDArray, Y: NDArray) -> float:

        mse = float(np.mean((Y-self.predict(X))**2))

        return mse
    
    def RMSE(self, X: NDArray, Y: NDArray) -> float:

        rmse = np.sqrt(
            self.MSE(X, Y)
        )
        
        return rmse
    
    def MAE(self, X: NDArray, Y: NDArray) -> float:

        mae = float(
            np.mean(
                np.abs(
            self.predict(X) - Y
        )))

        return mae
    
    def print_errors(self, X: NDArray, Y: NDArray) -> None:
        """
        Calculate and display common regression error metrics in a formatted table.

        Parameters:
        ----------
        X : NDArray
            Feature matrix of shape (n_samples, n_features) or (n_samples,).
        Y : NDArray
            True target values of shape (n_samples,) or (n_samples, 1).

        Prints:
        -------
        A table showing:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - MSE (Mean Squared Error)

        Example output:
        ----------------
        ╒═════════╤══════════╕
        │ Metric  │ Value    │
        ╞═════════╪══════════╡
        │ MAE     │ 2.1345   │
        │ RMSE    │ 3.1426   │
        │ MSE     │ 9.8764   │
        ╘═════════╧══════════╛
        """

        mae = self.MAE(X, Y)   
        rmse = self.RMSE(X, Y)  
        mse = self.MSE(X, Y)   

        table = [
            ['MAE', round(mae, 4)],
            ['RMSE', round(rmse, 4)],
            ['MSE', round(mse, 4)]
        ]

        print(tabulate(table, headers=['Metric', 'Value'], tablefmt='fancy_grid'))
    