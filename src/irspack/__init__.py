from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("irspack").version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass  # pragma: no cover

from .definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from .evaluation import *
from .recommenders import *
from .split import *
from .utils import *
