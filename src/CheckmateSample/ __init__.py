# __init__.py for CheckmateSample
from .generator import *

try:
    from ._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version("CheckmateSample")
    except PackageNotFoundError:
        __version__ = "unknown"

__all__ = ["__version__"]
