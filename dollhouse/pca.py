import numpy as np
from dollhouse.utils import normalize


class PrincipalComponentAnalysis:
    def __init__(self, top_components=1):
        self.top_components = top_components

    def fit(self, X):
        X = normalize(X, by_column=True)
        covariance_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = self.reorder_eigenvectors(
            covariance_matrix, self.top_components
        )
        self.explained_variance = self.explain_variance(eigenvalues)
        return self.change_basis(X, eigenvectors[:, : self.top_components])

    def reorder_eigenvectors(self, covariance_matrix, top_components):
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        reorder_descending = np.argsort(-eigenvalues)
        return eigenvalues[reorder_descending], eigenvectors[reorder_descending]

    def change_basis(self, X, top_eigenvectors):
        return X @ top_eigenvectors

    def explain_variance(self, eigenvalues):
        return eigenvalues / np.sum(eigenvalues)
