# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
A toolkit for managing extensions to the project msip.
"""

from pyiron_base import Toolkit, Project, JobFactoryCore
from pyiron_ising.model import Model, Chain1D, Square2D, Hex2D, FCC3D, BCC3D
from pyiron_ising.mutate import Flip, Swap, Cluster, Mutator
from pyiron_ising.job.ising import Ising
from pyiron_ising.job.parallel import ParallelIsing
from functools import wraps

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Oct 15, 2021"


class JobFactory(JobFactoryCore):
    @property
    def _job_class_dict(self) -> dict:
        return {
            'Ising': Ising,
            'ParallelIsing': ParallelIsing
        }


class ModelFactory:
    @classmethod
    @property
    def Model(cls):
        return Model

    @classmethod
    @property
    def Chain1D(cls):
        return Chain1D

    @classmethod
    @property
    def Square2D(cls):
        return Square2D

    @classmethod
    @property
    def Hex2D(cls):
        return Hex2D

    @classmethod
    @property
    def FCC3D(cls):
        return FCC3D

    @classmethod
    @property
    def BCC3D(cls):
        return BCC3D


class MutationFactory:
    @classmethod
    @property
    @wraps(Flip)
    def Flip(cls):
        return Flip

    @classmethod
    @property
    @wraps(Swap)
    def Swap(cls):
        return Swap

    @classmethod
    @property
    @wraps(Cluster)
    def Cluster(cls):
        return Cluster


class IsingTools(Toolkit):
    def __init__(self, project: Project):
        super().__init__(project)
        self._job = JobFactory(project)

    @property
    def job(self):
        return self._job

    @property
    def model(self):
        return ModelFactory

    @property
    def mutation(self):
        return MutationFactory

    @property
    @wraps(Mutator)
    def mutator(self):
        return Mutator
