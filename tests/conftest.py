from typing import Dict

import numpy as np
import pytest
import scipy.sparse as sps


@pytest.fixture()
def test_interaction_data() -> Dict[str, sps.csr_matrix]:
    X_small = sps.csr_matrix(
        np.asfarray(
            [[1, 1, 2, 3, 4], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
        )
    )
    return {"X_small": X_small}
