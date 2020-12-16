import numpy as np
import pytest
import scipy.sparse as sps
from irspack.recommenders import IALSRecommender

X_small = sps.csr_matrix(
    np.asfarray(
        [[1, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    )
)
X_many = np.random.rand(888, 512)
X_many[X_many <= 0.9] = 0
X_many[X_many > 0.9] = 1
X_many = sps.csr_matrix(X_many)
X_many.sort_indices()

X_many_dense = sps.csr_matrix(np.random.rand(133, 245))


@pytest.mark.parametrize(
    "X",
    [
        (X_small),
    ],
)
def test_ials_overfit(X):
    rec = IALSRecommender(X, n_components=3, alpha=0, reg=1e-10)
    X = X.toarray()
    rec.learn()
    reprod = rec.trainer.core_trainer.user.dot(rec.trainer.core_trainer.item.T)
    print(reprod)
    print(X)
    assert (((X - reprod) ** 2).mean() / (X ** 2).mean()) < 1e-5
