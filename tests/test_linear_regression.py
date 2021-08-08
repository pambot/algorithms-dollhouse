import numpy as np
from sklearn.datasets import make_regression
from numpy.testing import assert_allclose
from dollhouse.utils import train_test_split
from dollhouse.linear_regression import LinearRegression

np.random.seed(0)

def test_linear_regression():
    X, y, coefficients = make_regression(n_samples=1000, n_features=4, bias=0, noise=1, random_state=0, coef=True)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 990)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predicted = lr.predict(X_test)

    assert_allclose(y_predicted, y_test, rtol=0.1)
    assert_allclose(lr.coefficients, coefficients, rtol=0.01)
    

    