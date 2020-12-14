import unittest
from unittest.main import main
import numpy as np
import scipy.sparse as sps
from irspack.utils import rowwise_train_test_split


class TestSplit(unittest.TestCase):
    def setUp(self) -> None:
        self.ratings = sps.csr_matrix(
            np.asfarray([[1, 1, 2, 3], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
        )

    def test_split(self):
        X_1, X_2 = rowwise_train_test_split(self.ratings, test_ratio=0.5, random_seed=1)
        nnz_1 = (X_1.toarray() > 0).sum(axis=1)
        nnz_2 = (X_2.toarray() > 0).sum(axis=1)

        self.assertEqual(nnz_1[0] + nnz_2[0], 4)
        self.assertEqual(nnz_1[1] + nnz_2[1], 2)
        self.assertEqual(nnz_1[2] + nnz_2[2], 1)
        self.assertEqual(nnz_1[3] + nnz_2[3], 0)

        self.assertEqual(((self.ratings - X_1 - X_2).toarray() ** 2).sum(), 0)

        # should have no overwrap
        self.assertEqual((X_1.multiply(X_2).toarray() ** 2).sum(), 0)
