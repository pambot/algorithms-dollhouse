import numpy as np
from numpy.testing import assert_array_equal
from dollhouse.gaussian_mixtures import GaussianMixture1D


def test_gaussian_mixture_1d():
    np.random.seed(0)

    X = np.concatenate([
        np.random.normal(10, 1, 100),
        np.random.normal(1, 2, 100)
    ])

    y_expected = np.concatenate([
        np.zeros(100),
        np.ones(100)
    ])

    y_predicted = GaussianMixture1D(2).fit(X)
    assert_array_equal(y_predicted, y_expected)


