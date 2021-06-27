import numpy as np
from dollhouse.toolshed.optimizers import gradient_descent
from dollhouse.toolshed.statistics import normalize


class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=500):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X, y):
        X = normalize(X, by_column=True)
        self.n_samples, self.n_features = X.shape
        self.coefficients = np.zeros(self.n_features)
        self.coefficients = gradient_descent(
            X,
            y,
            self.coefficients,
            self.loss_function,
            self.gradient_function,
            self.learning_rate,
            self.max_iterations,
        )
        return

    def predict(self, X):
        X = normalize(X, by_column=True)
        return self.predict_function(X, self.coefficients)

    def loss_function(self, X, y, coefficients):
        return (
            np.sum((y - self.predict_function(X, coefficients)) ** 2) / self.n_samples
        )

    def gradient_function(self, X, y, coefficients):
        return -2 / self.n_samples * X.T @ (y - self.predict_function(X, coefficients))

    def predict_function(self, X, coefficients):
        return X @ coefficients
