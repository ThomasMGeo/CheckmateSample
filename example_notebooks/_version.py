# This is from MetPy! 

# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for versioning."""


from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("CheckmateSample")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"