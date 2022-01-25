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
    def __init__(self, weight: float = 1, imperfect_seeds_only: bool = False):
        super().__init__()
        self.storage.weight = weight
        self.storage.imperfect_seeds_only = imperfect_seeds_only

    @property
    def weight(self) -> float:
        return self.storage.weight

    @weight.setter
    def weight(self, w: float):
        self.storage.weight = w

    @property
    def imperfect_seeds_only(self) -> float:
        return self.storage.imperfect_seeds_only

    @imperfect_seeds_only.setter
    def imperfect_seeds_only(self, b: bool):
        self.storage.imperfect_seeds_only = b

    @abstractmethod
    def __call__(self, model: Model, *seed_sites: Union[int, None]) -> str and Union[int, Tuple]:
        """
        Given a model, updates it genome and returns a mutation identifier string and randomly chosen sites. Seed sites
        to target can be provided optionally, but otherwise must be generated randomly.
        """
        pass


class Flip(Mutation):
    """
    Change a spin to something else.

    __init__ args:
        weight (float): Relative selection probability weight when belonging to a mutator. (Default is 1.)
        imperfect_seeds_only (float): Only make initial site selection among sites with fitness less than 1. (Default
            is False, no constraints on site selection.)

    __call__ args:
        model (Model): The model to mutate.
        i (int|None): the index whose spin to flip. (Default is None, choose index randomly)
    """
    def __call__(self, model: Model, i: Union[int, None] = None) -> str and int:
        i = model.choose(1, forbid_perfect_sites=self.imperfect_seeds_only) if i is None else i
        other_spins = model.unique_spins[model.unique_spins != model.genome[i]]
        new_spin = np.random.choice(other_spins, 1, replace=False)
        model.genome[i] = new_spin
        return "flip", i


class Swap(Mutation):
    """
    Exchange two spins.

    __init__ args:
        weight (float): Relative selection probability weight when belonging to a mutator. (Default is 1.)
        imperfect_seeds_only (float): Only make initial site selection among sites with fitness less than 1. (Default
            is False, no constraints on site selection.)
        naive (bool): Choose indices at total random instead of ensuring they have different spins. (Default is False,
            make sure they have different spins first!)

    __call__ args:
        model (Model): The model to mutate.
        i, j (int|None): the indices whose spins to swap. (Default is None, choose indices (semi- depending on naive
            parameter) randomly.)
    """
    def __init__(self, weight: float = 1, imperfect_seeds_only: bool = False, naive: bool = False):
        super().__init__(weight=weight, imperfect_seeds_only=imperfect_seeds_only)
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
        i = model.choose(1, forbid_perfect_sites=self.imperfect_seeds_only) if i is None else i
        j = model.choose(
            1,
            forbidden_site=i if self.naive else None,
            forbidden_spin=model.genome[i] if not self.naive else None,
            forbid_perfect_sites=self.imperfect_seeds_only
        ) if j is None else j
        model.genome[[i, j]] = model.genome[[j, i]]
        return "naive_swap" if self.naive else "swap", i, j


class Cluster(Mutation):
    """
    Exchange two groups of spins.

    Uses breadth-first searching to construct two equally-sized clusters of like spins, where the larger group gets
    truncated down to match the size of the smaller. In case any cluster is size 1, the mutation registers as a "swap".

    Cluster construction by specifying the minimum and/or maximum neighbours with similar spin for a candidate in the
    breadth first search to qualify for joining the cluster.

    __init__ args:
        weight (float): Relative selection probability weight when belonging to a mutator. (Default is 1.)
        imperfect_seeds_only (float): Only make initial site selection among sites with fitness less than 1. (Default
            is False, no constraints on site selection.)
        min_like_neighbors (int|None): The minimum numbers of like-spin neighbours to qualify for addition to the
            cluster. (Default is None, no restriction.)
        max_like_neighbors (int|None): The maximum numbers of like-spin neighbours to qualify for addition to the
            cluster. (Default is None, no restriction.)
        max_size (int|None): The largest cluster size to allow.

    __call__ args:
        model (Model): The model to mutate.
        i, j (int|None): the indices whose spins to swap. (Default is None, choose indices (semi- depending on naive
            parameter) randomly.)
    """
    def __init__(
            self,
            weight: float = 1,
            imperfect_seeds_only: bool = False,
            min_like_neighbors: Union[int, None] = None,
            max_like_neighbors: Union[int, None] = None,
            max_size: Union[int, None] = None,
    ):
        super().__init__(weight=weight, imperfect_seeds_only=imperfect_seeds_only)
        self.storage.min_like_neighbors = min_like_neighbors
        self.storage.max_like_neighbors = max_like_neighbors
        self.storage.max_size = max_size

    @property
    def min_like_neighbors(self) -> int:
        return self.storage.min_like_neighbors

    @property
    def max_like_neighbors(self) -> int:
        return self.storage.max_like_neighbors

    @property
    def max_size(self) -> int:
        return self.storage.max_size

    @staticmethod
    def _neighbor_match_condition(
            i: int,
            j: int,
            min_like_neighbors: int,
            max_like_neighbors: int,
            match_genome: np.ndarray,
            match_topology: np.ndarray
    ) -> bool:
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
        i = model.choose(1, forbid_perfect_sites=self.imperfect_seeds_only) if i is None else i
        j = model.choose(
            1,
            forbidden_spin=model.genome[i],
            forbid_perfect_sites=self.imperfect_seeds_only
        ) if j is None else j
        cluster_i, cluster_j = double_bfs(
            i, j, model.topology, max_size=self.max_size,
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
            identifier += "_imp" if self.imperfect_seeds_only else ""
            identifier += "" if self.max_size is None else f"_ms{self.max_size}"
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

    @wraps(Flip)
    @append_to_mutator
    def Flip(self, weight: float = 1, imperfect_seeds_only: bool = False) -> Flip:
        return Flip(weight=weight, imperfect_seeds_only=imperfect_seeds_only)

    @wraps(Swap)
    @append_to_mutator
    def Swap(self, weight: float = 1, imperfect_seeds_only: bool = False, naive: bool = False) -> Swap:
        return Swap(weight=weight, imperfect_seeds_only=imperfect_seeds_only, naive=naive)

    @wraps(Cluster)
    @append_to_mutator
    def Cluster(
            self,
            weight: float = 1,
            imperfect_seeds_only: bool = False,
            min_like_neighbors: Union[int, None] = None,
            max_like_neighbors: Union[int, None] = None,
            max_size: Union[int, None] = None
    ) -> Cluster:
        return Cluster(
            weight=weight,
            imperfect_seeds_only=imperfect_seeds_only,
            min_like_neighbors=min_like_neighbors,
            max_like_neighbors=max_like_neighbors,
            max_size=max_size
        )

    @append_to_mutator
    def __call__(self, mutation):
        return mutation


class Mutator(HasStorage):
    """
    Modifies the genome of `Model` classes.

    On call, a mutation is selected by a (weighted) random process, the model genome is updated, and a string identifier
    for the applied mutation is returned.

    Attributes:
        mutations (list[Mutation]): A collection of mutations to choose from.
        add (MutationAdder): A helper property which lets you use tab-completion for adding mutations.
        normalized_weights (numpy.ndarray[float]): The normalized relative weights of each mutation.
    """
    def __init__(self):
        """A class for modifying the genome of `model` classes."""
        super().__init__()
        self.storage.mutations = DataContainer(table_name='mutations')
        self._adder = MutationAdder(self)

    @property
    def mutations(self) -> DataContainer:  # [Type[Mutation]]
        return self.storage.mutations

    @property
    def add(self) -> MutationAdder:
        """Add a mutation to the list of available mutations."""
        return self._adder

    def append(self, mutation: Mutation):
        self.mutations.append(mutation)

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
