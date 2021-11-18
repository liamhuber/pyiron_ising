__all__ = []
__version__ = '1.0'

from pyiron_base import Project
from pyiron_ising.toolkit import IsingTools
Project.register_tools('ising', IsingTools)
from pyiron_ising.model import Model, Chain1D, Square2D, Hex2D, BCC3D, FCC3D
from pyiron_ising.mutate import Mutation, Flip, Swap, Cluster, Mutator
