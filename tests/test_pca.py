import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from dollhouse.pca import PrincipalComponentAnalysis

np.random.seed(0)


def test_principal_component_analysis():
    X, y = load_iris(return_X_y=True)
    X = X[y == 2]

    pca = PrincipalComponentAnalysis(top_components=2)
    pca.fit(X)

    variance_expected = np.array([0.60512286, 0.25889329, 0.10507932, 0.03090453])
    assert_allclose(pca.explained_variance, variance_expected, rtol=0.01)
