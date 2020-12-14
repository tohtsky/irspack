import unittest
import numpy as np
import scipy.sparse as sps
from irspack.utils import sparse_mm_threaded


class TestSparseMatrixMultiplication(unittest.TestCase):
    def setUp(self) -> None:
        self.X = sps.csr_matrix(
            np.asfarray([[1, 1, 2, 3], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
        )

    def test_split(self):
        X_mm = sparse_mm_threaded(self.X, self.X.T, 4).toarray()
        X_sps = self.X.dot(self.X.T)
        X_diff = X_mm - X_sps
        self.assertTrue(np.all(X_diff == 0))