import numpy as np


def normalize(v, by_column=False):
    axis = 0 if by_column else None
    return (v - np.min(v, axis=axis)) / (np.max(v, axis=axis) - np.min(v, axis=axis))


def train_test_split(X, y, n_train):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]
