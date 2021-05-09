import numpy as np

from irspack.definitions import OptionalRandomState


def convert_randomstate(arg: OptionalRandomState) -> np.random.RandomState:
    if arg is None or isinstance(arg, int):
        return np.random.RandomState(arg)
    elif isinstance(arg, np.random.RandomState):
        return arg
    else:
        raise ValueError(f"{arg} cannot be interpreted as a random state.")
