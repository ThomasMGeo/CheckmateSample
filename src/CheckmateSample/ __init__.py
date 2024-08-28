# __init__.py for CheckmateSample
from importlib.metadata import version, PackageNotFoundError
from .generator import * 

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
