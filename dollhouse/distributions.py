import numpy as np


def gaussian(x, mean, standard_deviation):
    return (
        1
        / np.sqrt(2 * np.pi * standard_deviation ** 2)
        * np.exp(-1 / (2 * standard_deviation ** 2) * (x - mean) ** 2)
    )
