import numpy as np
from dollhouse.optimizers import gradient_descent


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=500):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X, y):
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
        return np.round(self.predict_function(X, self.coefficients))

    def loss_function(self, X, y, coefficients):
        y_predicted = self.predict_function(X, coefficients)
        return (
            -np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
            / self.n_samples
        )

    def gradient_function(self, X, y, coefficients):
        return X.T @ (self.predict_function(X, coefficients) - y) / self.n_samples

    def predict_function(self, X, coefficients):
        return 1 / (1 + np.exp(-(X @ coefficients)))
