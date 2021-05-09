from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("irspack").version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass  # pragma: no cover

from irspack.definitions import DenseScoreArray, InteractionMatrix, UserIndexArray
from irspack.evaluator import *
from irspack.optimizers import *
from irspack.recommenders import *
from irspack.split import *
from irspack.utils import *
