from solvers.grad_methods import GradientDescent, LBFGS
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

class SVMClassificator:
    def __init__(self,
                 n_iter: int = 1000,
                 lr: float = 0.0001,
                 C: float = 1.0,
                 m: int = 10,
                 kernel: str = "linear",
                 degree: int = 2,
                 r: float = 1.0,
                 gamma: float = 0.5,
                 tolerance: float = 1e-4) -> None:
        
        valid_kernels = {"linear", "sigmoid", "poly", "rbf"}
        if kernel not in valid_kernels:
            raise ValueError(f"kernel must be one of {valid_kernels}, got '{kernel}'")
        
        
        self.kernel = kernel
        self.n_iter = n_iter
        self.lr = lr
        self.C = C
        self.tolerance = tolerance
        self.m = m 
        
        self.degree = degree  
        self.r = r           
        self.gamma = gamma   
    def compute_kernel(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T
        elif self.kernel == "poly":
            return (X1 @ X2.T + self.r) ** self.degree
        elif self.kernel == "sigmoid":
            return np.tanh(self.gamma * (X1 @ X2.T) + self.r)
        elif self.kernel == "rbf":
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel")
    
    def fit(self,X,y):
        y = 2 * y - 1
        self.X = X
        self.y = y
        np.random.seed(42)
        
        self.alpha = np.random.random(X.shape[0])
        self.b = 0
        self.ones = np.ones(X.shape[0]) 

        y_mul_kernal = np.outer(y, y) * self.compute_kernel(X, X) 

        prev_loss = None
        tolerance = self.tolerance  

        for i in range(self.n_iter):
            gradient = self.ones - y_mul_kernal.dot(self.alpha)
            
            self.alpha += self.lr * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)
            
            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_mul_kernal)

            if prev_loss is not None and abs(loss - prev_loss) < tolerance:
                print(f"Converged at iteration {i}")
                break
            prev_loss = loss
      

        alpha_index = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        
        b_list = []        
        for index in alpha_index:
            b_list.append(y[index] - (self.alpha * y).dot(self.compute_kernel(X, X[index:index+1])))
        if len(b_list) == 0:
            self.b = 0
        else:
            self.b = np.mean(b_list)
        return self

    def predict(self, X):
        return (np.sign(self.decision_function(X)) + 1)/2
    
    def decision_function(self, X):
        return (self.alpha * self.y).dot(self.compute_kernel(self.X, X)) + self.b

    def plot(self,i, j ,title='Plot for non linear SVM'):
        plt.scatter(self.X[:, i], self.X[:, j], c=self.y, s=50, cmap='winter', alpha=.5)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 50)
        yy = np.linspace(ylim[0], ylim[1], 50)

        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        if xy.shape[1] != self.X.shape[1]:
            padded_xy = np.zeros((xy.shape[0], self.X.shape[1]))
            padded_xy[:, [i, j]] = xy
            xy = padded_xy
        Z = self.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, levels=[-1, 0, 1],linestyles=['--', '-', '--'])
        plt.title(title)
        plt.show()