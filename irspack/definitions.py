from typing import Any, List, Optional, Union

import numpy as np
from scipy import sparse as sps

InteractionMatrix = Union[sps.csr_matrix, sps.csc_matrix]
ProfileMatrix = Union[sps.csr_matrix, sps.csc_matrix, np.ndarray]

# wait until better numpy stub support
DenseScoreArray = np.ndarray
DenseMatrix = np.ndarray
UserIndexArray = np.ndarray
