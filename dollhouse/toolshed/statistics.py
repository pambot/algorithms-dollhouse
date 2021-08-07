import numpy as np


def normalize(v, by_column=False):
    axis = 0 if by_column else None
    return (v - np.mean(v, axis=axis)) / (np.max(v, axis=axis) - np.min(v, axis=axis))


def gaussian(x, mean, standard_deviation):
    return (
        1
        / np.sqrt(2 * np.pi * standard_deviation ** 2)
        * np.exp(-1 / (2 * standard_deviation ** 2) * (x - mean) ** 2)
    )
