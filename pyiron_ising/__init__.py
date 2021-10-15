__all__ = []

from pyiron_base import Project
from pyiron_ising.toolkit import IsingTools
Project.register_tools('ising', IsingTools)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
