import numpy as np
from dollhouse.utils import normalize


class PrincipalComponentAnalysis:
    def __init__(self, top_components=1):
        self.top_components = top_components

    def fit(self, X):
        X = normalize(X, by_column=True)
        covariance_matrix = np.cov(X)
        top_eigenvectors = self.prune_components(covariance_matrix, self.top_components)
        return self.change_basis(X, top_eigenvectors)

    def prune_components(self, covariance_matrix, top_components):
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        ordered_eigenvectors = eigenvectors[np.argsort(-eigenvalues)]
        return ordered_eigenvectors[:, :top_components]

    def change_basis(self, X, eigenvectors):
        return eigenvectors.T * X.T
