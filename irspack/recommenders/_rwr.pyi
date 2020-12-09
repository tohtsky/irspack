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
"""Backend C++ inplementation for Random walk with restart."""
from typing import *
from typing import Iterable as iterable
from typing import Iterator as iterator
from numpy import float32, float64, int32

_Shape = Tuple[int, ...]
import scipy.sparse

__all__ = ["RandomWalkGenerator"]


class RandomWalkGenerator:
    def __init__(self, arg0: scipy.sparse.csr_matrix[float32]) -> None:
        ...

    def run_with_fixed_step(
        self, arg0: int, arg1: int, arg2: int, arg3: int
    ) -> scipy.sparse.csr_matrix[int32]:
        ...

    def run_with_restart(
        self, arg0: float, arg1: int, arg2: int, arg3: int, arg4: int
    ) -> scipy.sparse.csr_matrix[int32]:
        ...

    pass
