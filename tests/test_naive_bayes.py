import numpy as np
from sklearn.datasets import make_classification
from dollhouse.utils import train_test_split, assert_array_mismatch_less_than
from dollhouse.naive_bayes import GaussianNaiveBayes

np.random.seed(0)


def test_naive_bayes():
    X, y = make_classification(
        n_samples=1000, n_features=4, class_sep=2, random_state=0
    )
    X_train, y_train, X_test, y_test = train_test_split(X, y, 990)

    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    y_predicted = nb.predict(X_test)

    assert_array_mismatch_less_than(y_predicted, y_test, tol=0.05)

    
