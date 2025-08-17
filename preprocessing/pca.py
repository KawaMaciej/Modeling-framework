import numpy as np

def PCA(X, k):
    X_centered = X - X.mean(axis=0)

    cov = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    X_pca_k = X_centered @ eigenvectors[:, :k]

    return X_pca_k, eigenvalues, eigenvectors