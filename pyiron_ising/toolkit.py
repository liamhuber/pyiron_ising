# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
A toolkit for managing extensions from ising.
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
    @property
    def Model(self):
        return Model

    @property
    def Chain1D(self):
        return Chain1D

    @property
    def Square2D(self):
        return Square2D

    @property
    def Hex2D(self):
        return Hex2D

    @property
    def FCC3D(self):
        return FCC3D

    @property
    def BCC3D(self):
        return BCC3D


class MutationFactory:
    @property
    def Flip(self):
        return Flip

    @property
    def Swap(self):
        return Swap

    @property
    def Cluster(self):
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
        return ModelFactory()

    @property
    def mutation(self):
        return MutationFactory()

    @property
    @wraps(Mutator)
    def mutator(self):
        return Mutator
