# __init__.py for CheckmateSample
from .generator import *

from ._version import get_version  # noqa: E402

__version__ = get_version()
del get_version
