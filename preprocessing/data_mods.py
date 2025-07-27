import numpy as np



def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
    X = np.array(X)
    y = np.array(y)
    
    assert len(X) == len(y), "X and y must be the same length"
    n_samples = len(X)

    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be float or int")

    indices = np.arange(n_samples)
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def CategoricalEncoder(X, labels):
    for label in labels:
        X[label] = X[label].astype("category").cat.codes
    return X[labels]

def StandarScaler(X, labels):
    for label in labels:
        X[label] = (X[label]-np.mean(X[label]))/np.std(X[label])

    return X[labels]