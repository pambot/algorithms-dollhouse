import numpy as np


def normalize(v, by_column=False):
    axis = 0 if by_column else None
    return (v - np.mean(v, axis=axis)) / (np.max(v, axis=axis) - np.min(v, axis=axis))
