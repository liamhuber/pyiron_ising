# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
A representation of the Ising model.
"""

from pyiron_base import HasStorage
from pyiron_atomistics import Atoms
from pyiron_atomistics.atomistics.structure.periodic_table import PeriodicTable
from pyiron_atomistics.atomistics.structure.factory import StructureFactory
from pyiron_atomistics.atomistics.structure.neighbors import Neighbors
import numpy as np
from abc import ABC, abstractmethod
from numbers import Integral
from typing import Union, List, Tuple
InteractionLike = Union[np.ndarray, List[List[Union[float, int]]], 'xenophilic', 'xenophobic']

__author__ = "Liam Huber, Vijay Bhuva"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Oct 15, 2021"


class Model(HasStorage):
    """
    Topology and spin interaction for an Ising model.

    Note: Since the model is built on top of the `Atoms` class, the dimensionality of models is restricted to <=3.
    """
    def __init__(
            self,
            structure: Atoms,
            n_neighbors: int,
            interaction: InteractionLike = 'xenophobic',
            shuffle: bool = True
    ):
        """
        An Ising model system.

        Args:
            structure (Atoms): The geometry of the model, defining lattice points (positions) and spin (
            n_neighbors (int): The number of closest neighbors to consider topologically connected for the Ising model.
            interaction (numpy.ndarray|list|'xenophilic'|'xenophobic'): A matrix of shape `(n_neighbors, n_neighbors)`
                defining the interaction of connected spins. 'xenophilic' gives -1 on the diagonal and +1 elsewhere
                (i.e. positive fitness for un-like neighbours, i.e. phase mixing), while 'xenophobic' gives the opposite
                (+1 on main diagonal, -1 elsewhere, i.e. max fitness for like-neighbors, i.e. full phase separation).
                (Default is 'xenophobic' -- phase separation.)
            shuffle (bool): Whether to shuffle the spins on initialization. (Default is True.)
        """
        super().__init__()
        self.storage.structure = None
        self.storage.n_neighbors = None
        self.storage.interaction = None
        self.structure = structure
        self.n_neighbors = n_neighbors
        self.interaction = interaction
        self._neighbors = None
        if shuffle:
            self.shuffle()

    @property
    def structure(self) -> Atoms:
        return self.storage.structure

    @structure.setter
    def structure(self, structure: Atoms):
        if not isinstance(structure, Atoms):
            raise TypeError(f'Expected a structure of type {Atoms.__name__} but got {type(structure)}')
        self.storage.structure = structure

    @property
    def n_neighbors(self) -> int:
        return self.storage.n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n: int):
        if not isinstance(n, np.int):
            raise TypeError(f'Expected an int but got {type(n)}')
        self.storage.n_neighbors = n

    @property
    def interaction(self) -> np.ndarray:
        return self.storage.interaction

    @interaction.setter
    def interaction(self, interaction: InteractionLike):
        if isinstance(interaction, str):
            interaction = interaction.lower()
            if interaction == 'xenophilic':
                interaction = np.ones((self.n_spins, self.n_spins)) - 2 * np.eye(self.n_spins)
            elif interaction == 'xenophobic':
                interaction = 2 * np.eye(self.n_spins) - np.ones((self.n_spins, self.n_spins))
            else:
                raise ValueError(f'For string-like interactions, please choose "xenophilic" or "xenophobic".')
        else:
            interaction = np.array(interaction)

        if not np.all(interaction.shape == (self.n_spins, self.n_spins)):
            raise ValueError(
                f'Expected interaction matrix to be square in number of spins, {self.n_spins}, but got shape '
                f'{interaction.shape}'
            )

        self.storage.interaction = interaction

    @property
    def neighbors(self) -> Neighbors:
        if self._neighbors is None:
            self._neighbors = self.structure.get_neighbors(num_neighbors=self.n_neighbors)
        return self._neighbors

    @property
    def topology(self) -> np.ndarray:
        return self.neighbors.indices

    @property
    def genome(self) -> np.ndarray:
        return self.structure.indices

    @genome.setter
    def genome(self, new_genome: Union[np.ndarray, List[int]]):
        self.structure.indices = new_genome

    def __len__(self) -> int:
        return len(self.structure)

    @property
    def n_spins(self) -> int:
        return self.structure.get_number_of_species()

    @property
    def unique_spins(self):
        return np.arange(self.n_spins)

    def shuffle(self):
        """Randomly reassign spins."""
        self.structure.indices = self.genome[self.choose(len(self))]

    def choose(
            self,
            n_choices: int,
            mask: Union[np.ndarray, None] = None,
            forbidden_site: Union[Integral, None] = None,
            forbidden_spin: Union[Integral, None] = None,
            forbid_perfect_sites: bool = False,
    ):
        """Choose sites randomly without replacement."""
        true = np.ones(len(self), dtype=bool)
        mask = (mask if mask is not None else true) * \
               (self.sites != forbidden_site if forbidden_site is not None else true) * \
               (self.genome != forbidden_spin if forbidden_spin is not None else true) * \
               (self.fitness_array < 1 if forbid_perfect_sites else true)
        sites = self.sites[mask] if mask is not None else self.sites
        choice = np.random.choice(sites, n_choices, replace=False)
        if n_choices == 1:
            return choice[0]
        else:
            return choice

    @property
    def sites(self) -> np.ndarray:
        return np.arange(len(self))

    def get_sites_by_spin(self, spin_value: int) -> np.ndarray:
        """Get all the site ids that have a particular spin value."""
        return self.sites[self.genome == spin_value]

    def plot3d(self, **kwargs):
        """Visualize the lattice coloured according to spin."""
        return self.structure.plot3d(**kwargs)

    def copy(self):
        """Deep copy."""
        return self.__class__(
            structure=self.structure.copy(),
            n_neighbors=self.n_neighbors,
            interaction=self.interaction,
            shuffle=False
        )

    @property
    def fitness_array(self):
        spins = self.genome
        neighbor_spins = self.genome[self.topology]
        return self.interaction[spins[:, np.newaxis], neighbor_spins].mean(axis=-1)

    @property
    def fitness(self):
        return self.fitness_array.mean()


class _SpecialModel(Model, ABC):
    """
    A special class for models with pre-defined topology.

    New children need to define the underlying unit structure, dimensionality, and number of neighbours. The unit
    structure gets repeated along the x/xy/xyz direction(s) for models in 1/2/3 dimensions.
    """
    def __init__(
            self,
            repetitions: Union[int, Tuple] = 2,
            n_spins: int = 2,
            interaction: InteractionLike = 'xenophobic',
            shuffle: bool = True,
            spin_fractions: Union[np.ndarray, List[float], None] = None,
            _structure: Union[None, Atoms] = None  # For internal use copying
    ):
        """
        Build a special model with pre-defined topology.

        Args:
            repetitions (int|tuple[int]): Repetitions of the special structure. (Default is 2.)
            n_spins (int): How many different spins to use. (Default is 2.)
            interaction (numpy.ndarray|list|'xenophilic'|'xenophobic'): A matrix of shape `(n_neighbors, n_neighbors)`
                defining the interaction of connected spins. 'xenophilic' gives -1 on the diagonal and +1 elsewhere
                (i.e. positive fitness for un-like neighbours, i.e. phase mixing), while 'xenophobic' gives the opposite
                (+1 on main diagonal, -1 elsewhere, i.e. max fitness for like-neighbors, i.e. full phase separation).
                (Default is 'xenophobic' -- phase separation.)
            shuffle (bool): Whether to randomize the initial placement of spins, otherwise they appear consecutively by
                structure index. (Default is True.)
            spin_fractions (numpy.ndarray|list): The fraction of each spin to use. (Default is None, which assigns spins
                as equally as possible.)
        """
        if _structure is None:
            structure = self._unit_structure.repeat(self._clean_repetitions(repetitions))
            self._check_n_spins_is_valid(n_spins, structure)
            spin_fractions = self._clean_spin_fractions(spin_fractions, n_spins)
            structure = self._set_spins(structure, spin_fractions)
        else:
            structure = _structure
        super().__init__(structure, self.n_neighbors, interaction=interaction, shuffle=shuffle)

    @property
    @abstractmethod
    def _unit_structure(self) -> Atoms:
        pass

    @property
    @abstractmethod
    def _dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def _n_neighbors(self) -> int:
        pass

    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n: int):
        if n != self._n_neighbors:
            raise ValueError(f'{self.__class__} can only have {self._n_neighbors} neighbors but got {n}')
        self.storage.n_neighbors = self._n_neighbors

    @property
    def _symbols(self) -> np.ndarray:
        # Relying on Atoms puts a limit on our number of spins, but honestly it's very high...
        return PeriodicTable().dataframe.index.values

    def _check_n_spins_is_valid(self, n_spins: int, structure: Atoms):
        n_sites = len(structure)
        if n_spins > len(self._symbols) or n_spins < 1 or n_spins > n_sites:
            raise ValueError(
                f"{n_spins} spins were requested, but the minimum is 1 and max is {len(self._symbols)} and must "
                f"be less than the number of sites, {n_sites}"
            )

    def _clean_repetitions(self, repetitions: Union[int, Tuple]) -> Tuple:
        """Ensures that repetitions have the right dimensionality."""
        msg = f"Expected repetitions to be an integer or tuple of {self._dimension} integers, but got {repetitions}"
        if isinstance(repetitions, Integral):
            return self._dimension * (repetitions,) + (3 - self._dimension) * (1,)
        elif hasattr(repetitions, '__len__') and len(repetitions) == self._dimension:
            return repetitions + (3 - self._dimension) * (1,)
        else:
            raise TypeError(msg)

    @staticmethod
    def _clean_spin_fractions(spin_fractions: Union[np.ndarray, List[float]], n_spins: int) -> np.ndarray:
        if spin_fractions is None:
            spin_fractions = np.array([1 / n_spins] * n_spins)
        elif len(spin_fractions) != n_spins or not np.isclose(np.sum(spin_fractions), 1):
            raise ValueError(f'Asked for {n_spins} spins summing to 1 but gave {len(spin_fractions)} spin fractions'
                             f'summing to {np.sum(spin_fractions)}.')
        return spin_fractions

    def _set_spins(self, structure: Atoms, spin_fractions: Union[np.ndarray, List[float], None]) -> Atoms:
        dividing_ids = (len(structure) * spin_fractions.cumsum()).astype(int)
        structure[:dividing_ids[0]] = self._symbols[0]
        for n, id_ in enumerate(dividing_ids[1:]):
            structure[dividing_ids[n]:id_] = self._symbols[n + 1]
        return structure

    def _int_to_dimensional_tuple(self, repetitions: Union[int, Tuple]) -> Tuple:
        if isinstance(int, Integral):
            repetitions = self._dimension * (repetitions,)
        return repetitions

    def copy(self):
        """Deep copy."""
        return self.__class__(
            _structure=self.structure.copy(),
            interaction=self.interaction,
            shuffle=False
        )


class Chain1D(_SpecialModel):
    @property
    def _unit_structure(self) -> Atoms:
        structure = StructureFactory().bulk("Ac", a=2, cubic=True, crystalstructure="fcc")
        del structure[[1, 2, 3]]
        structure.cell[1:, 1:] *= 10  # Expand lattice to avoid neighbours
        return structure

    @property
    def _dimension(self) -> int:
        return 1

    @property
    def _n_neighbors(self) -> int:
        return 2


class _Special2D(_SpecialModel, ABC):
    @property
    def _dimension(self) -> int:
        return 2


class Square2D(_Special2D):
    @property
    def _unit_structure(self) -> Atoms:
        structure = StructureFactory().bulk("Ac", a=2, cubic=True, crystalstructure="bcc")
        del structure[1]
        structure.cell[2, 2] *= 10
        return structure

    @property
    def _n_neighbors(self) -> int:
        return 4


class Hex2D(_Special2D):
    @property
    def _unit_structure(self) -> Atoms:
        structure = StructureFactory().bulk("Ac", a=2, c=2, orthorhombic=True, crystalstructure="hcp")
        top_layer = np.argwhere(structure.positions[:, 2] > 0.25 * structure.cell.array[2, 2]).flatten()
        del structure[top_layer]
        structure.positions[:, 2] = 0
        structure.cell[2, 2] *= 4
        return structure

    @property
    def _n_neighbors(self) -> int:
        return 6


class _Special3D(_SpecialModel, ABC):
    @property
    def _dimension(self) -> int:
        return 3


class BCC3D(_Special3D):
    @property
    def _unit_structure(self) -> Atoms:
        structure = StructureFactory().bulk("Ac", a=2, cubic=True, crystalstructure="bcc")
        return structure

    @property
    def _n_neighbors(self) -> int:
        return 8


class FCC3D(_Special3D):
    @property
    def _unit_structure(self) -> Atoms:
        structure = StructureFactory().bulk("Ac", a=3, cubic=True, crystalstructure="fcc")
        return structure

    @property
    def _n_neighbors(self) -> int:
        return 12
