import numpy as np


def normalize(v, by_column=False):
    axis = 0 if by_column else None
    return (v - np.mean(v, axis=axis)) / (np.max(v, axis=axis) - np.min(v, axis=axis))


def gaussian(x, mean, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-1/(2 * variance) * (x - mean) ** 2)


def multivariate_gaussian(X, means, covariance):
    k = X.shape[1]
    return 1/np.sqrt((2 * np.pi)**k * np.linalg.det(covariance)) * np.exp(-1/2 * np.linalg.inv(covariance) * (X - means).T * (X - means))