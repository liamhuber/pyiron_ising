__all__ = []
__version__ = '1.0'

from pyiron_base import Project
from pyiron_ising.toolkit import IsingTools
Project.register_tools('ising', IsingTools)
