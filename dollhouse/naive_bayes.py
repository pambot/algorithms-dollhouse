import numpy as np
from dollhouse.distributions import gaussian


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.array(sorted(set(y)))
        self.n_classes = len(self.classes)
        n_samples, self.n_features = X.shape

        self.P_y = self.calculate_priors(y)
        self.means, self.standard_deviations = self.estimate_class_parameters(X, y)
        return

    def predict(self, X):
        n_samples = X.shape[0]
        P_y_given_X = np.zeros((n_samples, self.n_classes))
        for i, _ in enumerate(self.classes):
            P_y_given_X[:, i] = np.log(
                self.P_y[i]
            ) + self.joint_log_likelihood_P_X_given_y(X, i)
        return self.classes[np.argmax(P_y_given_X, axis=1)]

    def calculate_priors(self, y):
        P_y = np.zeros(self.n_classes)
        for i, c in enumerate(self.classes):
            P_y[i] = len(y[y == c]) / len(y)
        return P_y

    def estimate_class_parameters(self, X, y):
        def estimate_means(X, y):
            means = np.zeros((self.n_classes, self.n_features))
            for i, c in enumerate(self.classes):
                means[i, :] = np.mean(X[y == c, :], axis=0)
            return means

        def estimate_standard_deviations(X, y, means):
            standard_deviations = np.zeros((self.n_classes, self.n_features))
            for i, c in enumerate(self.classes):
                standard_deviations[i, :] = np.sqrt(
                    np.sum((X[y == c, :] - means[i, :]) ** 2, axis=0)
                    / X[y == c, :].shape[0]
                )
            return standard_deviations

        means = estimate_means(X, y)
        standard_deviations = estimate_standard_deviations(X, y, means)
        return means, standard_deviations

    def joint_log_likelihood_P_X_given_y(self, X, i):
        return np.sum(
            np.log(gaussian(X, self.means[i, :], self.standard_deviations[i, :])),
            axis=1,
        )
