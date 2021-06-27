import numpy as np


def normalize(v, by_column=False):
    axis = 0 if by_column else None
    return (v - np.mean(v, axis=axis)) / (np.max(v, axis=axis) - np.min(v, axis=axis))


def gaussian(x, mean, standard_deviation):
    variance = standard_deviation ** 2
    return np.exp(-((x - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
