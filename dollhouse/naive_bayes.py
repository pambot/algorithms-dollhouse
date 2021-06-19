import numpy as np
from dollhouse.statistics import gaussian


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.array(sorted(set(y)))
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.P_y = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.standard_deviations = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes):
            # The priors of the classes are the class proportions
            self.P_y[i] = len(y[y == c]) / len(y)

            X_c = X[y == c, :]
            X_c_means = np.mean(X_c, axis=0)
            n_X_c = X_c.shape[0]

            # These parameter estimations are derived from MLE
            # They end up being the sample mean and biased sample standard deviations
            self.means[i, :] = X_c_means
            self.standard_deviations[i, :] = np.sqrt(
                np.sum((X_c - X_c_means) ** 2, axis=0) / n_X_c
            )
        return

    def predict(self, X):
        n_classes = len(self.classes)
        n_samples = X.shape[0]
        P_y_given_X = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes):
            class_means = self.means[i, :]
            class_standard_deviations = self.standard_deviations[i, :]
            P_y_given_X[:, i] = np.log(self.P_y[i]) + np.sum(
                np.log(gaussian(X, class_means, class_standard_deviations)), axis=1
            )
        return self.classes[np.argmax(P_y_given_X, axis=1)]
