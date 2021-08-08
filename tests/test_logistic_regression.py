import numpy as np
from sklearn.datasets import make_classification
from numpy.testing import assert_array_equal
from dollhouse.utils import train_test_split
from dollhouse.logistic_regression import LogisticRegression

np.random.seed(0)


def test_logistic_regression():
    X, y = make_classification(
        n_samples=1000, n_features=4, class_sep=2, random_state=0
    )
    X_train, y_train, X_test, y_test = train_test_split(X, y, 990)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_predicted = lr.predict(X_test)

    assert_array_equal(y_predicted, y_test)
