
import numpy as np
from numpy.typing import NDArray


def confusion_matrix(Y: np.ndarray, pred: np.ndarray, n_classes: int = None):
    if n_classes is None:
        n_classes = len(np.unique(Y))
    cm = np.zeros([n_classes,n_classes])

    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                cm[i,i] = np.sum((pred == np.unique(Y)[i]) & (Y == np.unique(Y)[i]))
            else:
                cm[i,j] = np.sum((pred == np.unique(Y)[j]) & (Y == np.unique(Y)[i]))
        
    return cm
    
def precision(Y: NDArray, pred: NDArray):
    cm = confusion_matrix(Y, pred)
    precision = []
    n_classes = cm.shape[0]
    for i in range(n_classes):
        denom = np.sum(cm[:, i])
        if denom == 0:
            precision.append(0.0)
        else:
            precision.append(cm[i, i] / denom)
    return np.array(precision)
    
def recall(Y: NDArray, pred: NDArray):
    cm = confusion_matrix(Y, pred)
    n_classes = cm.shape[0]
    recall = []
    for i in range(n_classes):
        denom = np.sum(cm[i, :])
        if denom == 0:
            recall.append(0.0)
        else:
            recall.append(cm[i, i] / denom)
    return np.array(recall)
    
def f1_stat(Y: NDArray, pred: NDArray):
    prec = precision(Y, pred)
    rec = recall(Y, pred)
    f1 = np.zeros_like(prec)
    for i in range(len(prec)):
        if prec[i] + rec[i] == 0:
            f1[i] = 0.0
        else:
            f1[i] = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
    return f1
    
def accuracy(Y: NDArray, pred: NDArray):
    cm = confusion_matrix(Y, pred)
    acc = np.trace(cm)/Y.shape[0]

    return np.array(acc)
    
def balanced_accuracy(Y: NDArray, pred: NDArray):
    cm = confusion_matrix(Y, pred)
    bal_acc = np.mean(recall(Y, pred))

    return np.array(bal_acc)
    
def NPV(Y: NDArray, pred: NDArray):
    cm = confusion_matrix(Y, pred)
    npv_per_class = []
    n_classes = cm.shape[0]
    for i in range(n_classes):
        FN = np.sum(cm[i, :]) - cm[i, i]
        mask = np.ones_like(cm, dtype=bool)
        mask[i, :] = False
        mask[:, i] = False
        TN = np.sum(cm[mask])
        denom = TN + FN
        npv = TN / denom if denom != 0 else 0.0
        npv_per_class.append(npv)
    return np.array(npv_per_class)

def FOR(Y:NDArray, pred: NDArray):
    return 1 - NPV(Y, pred)

def Fowlkes_Mallows(Y: NDArray, pred: NDArray):
    return np.sqrt(precision(Y, pred) * recall(Y, pred))