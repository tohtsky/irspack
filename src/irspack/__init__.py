from importlib.metadata import PackageNotFoundError, version

try:
    from ._version import __version__
except PackageNotFoundError:  # pragma: no cover
    ___version__ = "0.0.0"

from .definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from .evaluation import *
from .recommenders import *
from .split import *
from .utils import *
