# __init__.py for CheckmateSample
from .generator import *

from ._version import get_version  # noqa: E402
from .xarray import *  # noqa: F401, F403, E402

__version__ = get_version()
del get_version
