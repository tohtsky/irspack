try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("irspack")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass  # pragma: no cover

from .definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from .evaluation import *
from .recommenders import *
from .split import *
from .utils import *
