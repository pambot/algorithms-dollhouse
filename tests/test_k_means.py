import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import make_blobs
from dollhouse.k_means import KMeans

np.random.seed(0)


def test_k_means():
    X, y_expected = make_blobs(
        100, n_features=4, centers=2, cluster_std=0.5, random_state=0
    )

    y_predicted = KMeans(n_clusters=2).fit(X)

    assert_array_equal(y_predicted, y_expected)
