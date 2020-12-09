from typing import List, Any
from abc import ABC, abstractmethod
from scipy import sparse as sps
from scipy.sparse.csr import csr_matrix


class BaseEncoder(ABC):
    names: List[Any]

    def __init___(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform_sparse(self, arg: List[Any]) -> sps.csr_matrix:
        raise NotImplementedError("not implemented")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("not implemented")

