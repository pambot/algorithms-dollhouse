import numpy as np
from dollhouse.distributions import gaussian


class GaussianMixture1D:
    def __init__(self, n_clusters, max_iterations=500):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit(self, X):
        self.check_X(X)
        self.n_samples = len(X)
        means = np.linspace(np.min(X), np.max(X), self.n_clusters)
        standard_deviations = np.ones(self.n_clusters)
        sample_clusters = None
        for _ in range(self.max_iterations):
            sample_clusters, P_y_given_X = self.estimate_cluster_belonging(
                X, means, standard_deviations, sample_clusters
            )
            means, standard_deviations = self.calculate_new_cluster_parameters(
                X, P_y_given_X
            )
        sample_clusters, _ = self.estimate_cluster_belonging(
            X, means, standard_deviations, sample_clusters
        )
        return sample_clusters

    def estimate_cluster_belonging(
        self, X, means, standard_deviations, sample_clusters
    ):
        P_y = self.calculate_priors(sample_clusters)
        P_X_given_y = self.calculate_P_X_given_y(X, means, standard_deviations)
        P_y_given_X = (P_y * P_X_given_y).T / np.sum(P_y * P_X_given_y, axis=1)
        return np.argmin(P_y_given_X, axis=0), P_y_given_X

    def calculate_priors(self, sample_clusters):
        P_y = np.zeros(self.n_clusters)
        if not isinstance(sample_clusters, np.ndarray):
            for c in range(self.n_clusters):
                P_y[c] = 1 / self.n_clusters
        else:
            for c in range(self.n_clusters):
                P_y[c] = sample_clusters[sample_clusters == c].shape[0] / self.n_samples
        return P_y

    def calculate_P_X_given_y(self, X, means, standard_deviations):
        P_X_given_y = np.zeros((self.n_samples, self.n_clusters))
        for c in range(self.n_clusters):
            P_X_given_y[:, c] = gaussian(X, means[c], standard_deviations[c])
        return P_X_given_y

    def calculate_new_cluster_parameters(self, X, P_y_given_X):
        means = np.zeros(self.n_clusters)
        standard_deviations = np.zeros(self.n_clusters)
        for c in range(self.n_clusters):
            P_y_given_Xc = P_y_given_X[c, :]
            means[c] = np.sum((X * P_y_given_Xc)) / np.sum(P_y_given_Xc)
            standard_deviations[c] = np.sum(
                np.sqrt((P_y_given_Xc * (X - means[c]) ** 2)) / np.sum(P_y_given_Xc)
            )
        return means, standard_deviations

    def check_X(self, X):
        if len(X.shape) != 1:
            raise ValueError("Input array X must be 1D.")
