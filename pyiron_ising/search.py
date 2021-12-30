# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import List, Tuple, Callable, Union
import numpy as np
from pandas import unique  # Preserves order, and is apparently also faster than numpy's (probably because skips sort)

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


def _add_to_cluster(
        queue: List,
        cluster: List,
        topology: List[List],
        max_size: Union[int, None],
        condition_fnc: Callable,
        **condition_kwargs
) -> Tuple[List, List]:
    i = queue.pop()
    to_add = [j for j in topology[i] if condition_fnc(i, j, **condition_kwargs)]
    to_add = to_add[:max(max_size - len(cluster), 0)] if max_size is not None else to_add
    queue += np.setdiff1d(to_add, cluster).astype(int).tolist()
    cluster = unique(cluster + to_add).astype(int).tolist()
    return queue, cluster


def bfs(
        i: int,
        topology: Union[np.ndarray, List[List]],
        condition_fnc: Callable,
        max_size: Union[int, None] = None,
        **condition_kwargs
) -> List:
    """
        Breadth first search building a cluster starting at one node and obeying a condition function for adding new
        nodes.

        Args:
            i (int): Which node to start the search from.
            topology (numpy.ndarray | list): Per-site list of lists giving the all neighbouring nodes (i.e. a
                `Neighbors.indices` object).
            condition_fnc (fnc): A function for evaluating whether or not connected nodes should be added.
            max_size (int|None): Maximum cluster size. (Default is None, no limit.)
            *condition_args: Additional arguments for the condition function.

        Returns:
            (numpy.ndarray): The cluster built from the requested node obeying the condition function.
        """
    cluster = [i]
    queue = [i] if condition_fnc(i, i, **condition_kwargs) else []
    while queue:
        queue, cluster = _add_to_cluster(queue, cluster, topology, max_size, condition_fnc, **condition_kwargs)
    return cluster


def double_bfs(
        i: int,
        j: int,
        topology: Union[np.ndarray, List[List]],
        condition_fnc: callable,
        max_size: Union[int, None] = None,
        **condition_kwargs
) -> Tuple[List, List]:
    """
    Breadth first search building two clusters starting at two different nodes, obeying a condition function for
    adding new nodes, terminating when either search finishes, and truncating the larger cluster.

    Args:
        i, j (int): Which nodes to start the search from.
        topology (numpy.ndarray | list): Per-site list of lists giving the all neighbouring nodes (i.e. a
            `Neighbors.indices` object).
        condition_fnc (fnc): A function for evaluating whether or not connected nodes should be added.
        max_size (int|None): Maximum cluster size. (Default is None, no limit.)
        *condition_kwargs: Additional arguments for the condition function.

    Returns:
        (numpy.ndarray): The clusters of equal size built from the requested nodes obeying the condition function.
    """
    cluster1 = [i]
    cluster2 = [j]

    queue1 = [i] if condition_fnc(i, i, **condition_kwargs) else []
    queue2 = [j] if condition_fnc(j, j, **condition_kwargs) else []

    while queue1 and queue2:
        queue1, cluster1 = _add_to_cluster(queue1, cluster1, topology, max_size, condition_fnc, **condition_kwargs)
        queue2, cluster2 = _add_to_cluster(queue2, cluster2, topology, max_size, condition_fnc, **condition_kwargs)

    n_smallest = min(len(cluster1), len(cluster2))

    return cluster1[:n_smallest], cluster2[:n_smallest]
