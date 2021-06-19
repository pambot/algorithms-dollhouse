import numpy as np
from dollhouse.optimizers import gradient_descent


class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=500):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def loss_function(self, X, y, coefficients):
        n_samples = X.shape[0]
        return np.sum((y - self.predict_function(X, coefficients)) ** 2) / n_samples

    def gradient(self, X, y, coefficients):
        n_samples = X.shape[0]
        return -2 / n_samples * X.T @ (y - self.predict_function(X, coefficients))

    def fit(self, X, y):
        self.coefficients = np.zeros(X.shape[1])
        self.coefficients = gradient_descent(
            X,
            y,
            self.coefficients,
            self.loss_function,
            self.gradient,
            self.learning_rate,
            self.max_iterations,
        )
        return

    def predict_function(self, X, coefficients):
        return X @ coefficients

    def predict(self, X):
        return self.predict_function(X, self.coefficients)
