"""
Copyright 2020 BizReach, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float64, int32, float32

import scipy.sparse

m: int
n: int

__all__ = [
    "rowwise_train_test_split_d",
    "rowwise_train_test_split_f",
    "rowwise_train_test_split_i",
    "sparse_mm_threaded",
]


def rowwise_train_test_split_d(
    arg0: scipy.sparse.csr_matrix[float64], arg1: float, arg2: int
) -> Tuple[scipy.sparse.csc_matrix[float64], scipy.sparse.csr_matrix[float64]]:
    pass


def rowwise_train_test_split_f(
    arg0: scipy.sparse.csr_matrix[float32], arg1: float, arg2: int
) -> Tuple[scipy.sparse.csc_matrix[float32], scipy.sparse.csr_matrix[float32]]:
    pass


def rowwise_train_test_split_i(
    arg0: scipy.sparse.csr_matrix[float32], arg1: float, arg2: int
) -> Tuple[scipy.sparse.csc_matrix[float32], scipy.sparse.csr_matrix[float32]]:
    pass


def sparse_mm_threaded(
    arg0: scipy.sparse.csr_matrix[float64],
    arg1: scipy.sparse.csc_matrix[float64],
    arg2: int,
) -> scipy.sparse.csr_matrix[float64]:
    pass

