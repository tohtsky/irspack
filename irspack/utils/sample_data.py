import scipy.sparse as sps
from scipy.special import expit

from irspack.definitions import OptionalRandomState
from irspack.utils.random import convert_randomstate


def mf_example_data(
    n_users: int = 100,
    n_items: int = 100,
    n_components: int = 5,
    random_state: OptionalRandomState = None,
) -> sps.csr_matrix:
    rns = convert_randomstate(random_state)
    user_factors = rns.randn(n_users, n_components) / (n_components**0.5)
    item_factors = rns.randn(n_items, n_components) / (n_components**0.5)
    score = expit(user_factors.dot(item_factors.T))
    return sps.csr_matrix(rns.binomial(1, score))
