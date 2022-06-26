import os
from typing import Optional


def get_n_threads(n_threads: Optional[int]) -> int:
    if n_threads is not None:
        return n_threads
    else:
        try:
            n_threads_cand = os.environ.get(
                "IRSPACK_NUM_THREADS_DEFAULT", os.cpu_count()
            )
            return int(n_threads_cand or 1)
        except:
            raise ValueError(
                'failed to interpret "IRSPACK_NUM_THREADS_DEFAULT" as an integer.'
            )
