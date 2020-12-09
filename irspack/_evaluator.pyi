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
from numpy import float64, float32, flags

import numpy
import scipy.sparse

m: int
n: int

__all__ = ["EvaluatorCore", "Metrics"]


class EvaluatorCore:
    def __init__(self, arg0: scipy.sparse.csr_matrix[float64]) -> None:
        ...

    def get_metrics(
        self,
        arg0: numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous],
        arg1: int,
        arg2: int,
        arg3: int,
        arg4: bool,
    ) -> Metrics:
        ...

    pass


class Metrics:
    def __init__(self, arg0: int) -> None:
        ...

    def as_dict(self) -> Dict[str, float]:
        ...

    def merge(self, arg0: Metrics) -> None:
        ...

    pass

