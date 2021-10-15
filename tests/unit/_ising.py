# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import PyironTestCase
from abc import ABC, abstractmethod

# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import TestWithProject
# You would think PyironTestCase is sufficient, but this leaves log detritus lying about
from abc import ABC
from pyiron_atomistics.toolkit import StructureFactory
from pyiron_ising.model import Model


class TestIsing(TestWithProject, ABC):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        structure = StructureFactory().bulk("Al", cubic=True)
        structure[1] = "Cu"
        structure[2] = "Ag"
        structure[3] = "Au"
        cls.structure = structure
        cls.interaction = 'xenophilic'
        cls.n_neighbors = 12

    def setUp(self):
        super().setUp()
        self.model = Model(self.structure, self.n_neighbors, self.interaction, shuffle=False)
