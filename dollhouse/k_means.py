import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iterations=500):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iterations):
            sample_clusters = self.assign_samples_to_clusters(X, self.centroids)
            self.centroids = self.calculate_new_centroids(
                X, self.centroids, sample_clusters
            )
        return self.assign_samples_to_clusters(X, self.centroids)

    def initialize_centroids(self, X):
        random_samples = np.random.choice(
            self.n_samples, size=self.n_clusters, replace=False
        )
        return X[random_samples, :]

    def assign_samples_to_clusters(self, X, centroids):
        centroid_distances = np.zeros((self.n_samples, self.n_clusters))
        for c in range(self.n_clusters):
            centroid = centroids[c, :]
            centroid_distances[:, c] = self.distance_function(X, centroid)
        return np.argmin(centroid_distances, axis=1)

    def distance_function(self, X, centroid):
        return ((X - centroid) ** 2).sum(axis=1) ** 0.5

    def calculate_new_centroids(self, X, centroids, sample_clusters):
        for c in range(self.n_clusters):
            centroids[c, :] = X[sample_clusters == c, :].mean(axis=0)
        return centroids
