# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
Modify the genome of a model.
"""

from pyiron_base import HasStorage, DataContainer
from pyiron_ising.model import Model
from pyiron_ising.search import double_bfs
from typing import Type, Union, Callable, Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from functools import wraps

__author__ = "Liam Huber, Vijay Bhuva"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Oct 16, 2021"


class Mutation(HasStorage, ABC):
    def __init__(self, weight: float = 1):
        super().__init__()
        self.storage.weight = weight

    @property
    def weight(self) -> float:
        return self.storage.weight

    @weight.setter
    def weight(self, w: float):
        self.storage.weight = w

    @abstractmethod
    def __call__(self, model: Model, *seed_sites: Union[int, None]) -> str and Union[int, Tuple]:
        """
        Given a model, updates it genome and returns a mutation identifier string and randomly chosen sites. Seed sites
        to target can be provided optionally, but otherwise must be generated randomly.
        """
        pass


class Flip(Mutation):
    """Change a spin to something else."""
    def __call__(self, model: Model, i: Union[int, None] = None) -> str and int:
        i = model.choose(1) if i is None else i
        new_spin = model.choose(1, array=model.unique_spins, mask=model.unique_spins != model.genome[i])
        model.genome[i] = new_spin
        return "flip", i


class Swap(Mutation):
    """Exchange two spins."""
    def __init__(self, weight: float = 1, naive: bool = False):
        super().__init__(weight=weight)
        self.storage.naive = naive

    @property
    def naive(self) -> bool:
        return self.storage.naive

    def __call__(
            self,
            model: Type[Model],
            i: Union[int, None] = None,
            j: Union[int, None] = None
    ) -> str and int and int:
        i = model.choose(1) if i is None else i
        j = model.choose(1, mask=model.sites != i if self.naive else model.genome != model.genome[i]) if j is None \
            else j
        model.genome[[i, j]] = model.genome[[j, i]]
        return "naive_swap" if self.naive else "swap", i, j


class Cluster(Mutation):
    """Exchange two groups of spins."""
    def __init__(
            self,
            weight: float = 1,
            min_like_neighbors: Union[int, None] = None,
            max_like_neighbors: Union[int, None] = None
    ):
        super().__init__(weight=weight)
        self.storage.min_like_neighbors = min_like_neighbors
        self.storage.max_like_neighbors = max_like_neighbors

    @property
    def min_like_neighbors(self) -> int:
        return self.storage.min_like_neighbors

    @property
    def max_like_neighbors(self) -> int:
        return self.storage.max_like_neighbors

    @staticmethod
    def _neighbor_match_condition(i, j, min_like_neighbors, max_like_neighbors, match_genome, match_topology) -> bool:
        if match_genome[j] != match_genome[i]:
            return False
        elif min_like_neighbors is None and max_like_neighbors is None:
            return True
        else:
            n_like_neighbors = np.sum(match_genome[match_topology[j]] == match_genome[j])
            min_ok = min_like_neighbors is None or n_like_neighbors >= min_like_neighbors
            max_ok = max_like_neighbors is None or n_like_neighbors <= max_like_neighbors
            return max_ok and min_ok

    def __call__(
            self,
            model: Type[Model],
            i: Union[int, None] = None,
            j: Union[int, None] = None
    ) -> str and int and int:
        i = model.choose(1) if i is None else i
        j = model.choose(1, mask=model.genome != model.genome[i]) if j is None else j
        cluster_i, cluster_j = double_bfs(
            i, j, model.topology,
            condition_fnc=self._neighbor_match_condition,
            min_like_neighbors=self.min_like_neighbors,
            max_like_neighbors=self.max_like_neighbors,
            match_genome=model.genome,
            match_topology=model.topology
        )

        if len(cluster_i) == 1:
            return Swap()(model, i, j)
        else:
            model.genome[cluster_i + cluster_j] = model.genome[cluster_j + cluster_i]
            identifier = "cluster"
            identifier += "" if self.min_like_neighbors is None else f"_min{self.min_like_neighbors}"
            identifier += "" if self.max_like_neighbors is None else f"_max{self.max_like_neighbors}"
            return identifier, i, j


def append_to_mutator(method: Callable):
    def wrapper(self, *args, **kwargs):
        mutation = method(self, *args, **kwargs)
        self._mutator.mutations.append(mutation)
        return mutation
    return wrapper


class MutationAdder:
    # TODO: Find a more elegant solution that doesn't involve duplicating the signatures every time

    def __init__(self, mutator):
        self._mutator = mutator

    @wraps(Flip.__init__)
    @append_to_mutator
    def Flip(self, weight: float = 1) -> Flip:
        return Flip(weight=weight)

    @wraps(Swap.__init__)
    @append_to_mutator
    def Swap(self, weight: float = 1, naive: bool = False) -> Swap:
        return Swap(weight=weight, naive=naive)

    @wraps(Cluster.__init__)
    @append_to_mutator
    def Cluster(
            self,
            weight: float = 1,
            min_like_neighbors: Union[int, None] = None,
            max_like_neighbors: Union[int, None] = None
    ) -> Cluster:
        return Cluster(weight=weight, min_like_neighbors=min_like_neighbors, max_like_neighbors=max_like_neighbors)


class Mutator(HasStorage):
    """
    Modifies the genome of `Model` classes.

    On call, a mutation is selected by a (weighted) random process, the model genome is updated, and a string identifier
    for the applied mutation is returned.

    Attributes:
        mutations (list[Mutation]): A collection of mutations to choose from.
    """
    def __init__(self):
        """A class for modifying the genome of `model` classes."""
        super().__init__()
        self.storage.mutations = DataContainer(table_name='mutations')
        self._adder = MutationAdder(self)

    @property
    def mutations(self) -> DataContainer[Type[Mutation]]:
        return self.storage.mutations

    @property
    def add(self) -> MutationAdder:
        """Add a mutation to the list of available mutations."""
        return self._adder

    def __call__(self, model: Model) -> str and Union[int, Tuple]:
        """
        a mutation is selected by a (weighted) random process, the model genome is updated, and a string identifier
        for the applied mutation is returned.
        """
        mutation = np.random.choice(list(self.mutations.values()), size=1, p=self.normalized_weights)[0]
        return mutation(model)

    @property
    def normalized_weights(self) -> np.ndarray:
        weights = np.array([m.weight for m in self.mutations.values()]).astype(float)
        total = weights.sum()
        if np.isclose(total, 0):
            raise ValueError("The sum of all mutation rates was not significantly different from zero")
        return weights / total

    def __len__(self):
        return len(self.mutations)
