import scipy.sparse as sps
from scipy.special import expit

from ..definitions import OptionalRandomState
from .random import convert_randomstate


def mf_example_data(
    n_users: int = 100,
    n_items: int = 100,
    n_components: int = 5,
    random_state: OptionalRandomState = None,
    density_target: float = 0.3,
) -> sps.csr_matrix:
    rns = convert_randomstate(random_state)
    user_factors = rns.randn(n_users, n_components) / (n_components**0.5)
    item_factors = rns.randn(n_items, n_components) / (n_components**0.5)
    eps = 1e-5
    bias_max = 100.0
    bias_min = -100.0
    bias_try = 0.0
    all_scores = user_factors.dot(item_factors.T)
    for _ in range(100):
        expected_density = expit(all_scores.ravel() + bias_try).mean()
        if expected_density > density_target:
            bias_max = bias_try
        else:
            bias_min = bias_try
        if (bias_max - bias_min) < eps:
            break
        bias_try = (bias_max + bias_min) / 2.0

    score = expit(user_factors.dot(item_factors.T) + bias_try)
    return sps.csr_matrix(rns.binomial(1, score))
