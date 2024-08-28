# __init__.py for checkmate-sample
from importlib.metadata import version, PackageNotFoundError
from .generator import * 

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
