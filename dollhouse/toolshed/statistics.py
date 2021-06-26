import numpy as np


def normalize(v):
    return (v - np.max(v)) / (np.max(v) - np.min(v))


def gaussian(x, mean, standard_deviation):
    variance = standard_deviation ** 2
    return np.exp(-((x - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
