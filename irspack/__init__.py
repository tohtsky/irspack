# __version__ = "0.1.0.dev3"
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("irspack").version
except DistributionNotFound:
    # package is not installed
    pass
