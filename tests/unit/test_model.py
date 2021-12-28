# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from tests.unit._ising import TestIsing
from pyiron_ising.model import Model, Chain1D, Square2D, Hex2D, BCC3D, FCC3D
import numpy as np
from abc import ABC, abstractmethod
from numbers import Integral


class TestModel(TestIsing):

    def test_init(self):
        with self.assertRaises(TypeError):
            Model("not Atoms", self.n_neighbors, self.interaction)
        with self.assertRaises(ValueError):
            Model(self.structure, self.n_neighbors, "bad string")
        with self.assertRaises(ValueError):
            Model(self.structure, self.n_neighbors, np.ones(42))  # Wrong shape
        with self.assertRaises(ValueError):
            Model(self.structure, self.n_neighbors, {})  # Wrong type entirely
        with self.assertRaises(TypeError):
            Model(self.structure, "not an int", self.interaction)

    def test_copy(self):
        new_model = self.model.copy()
        self.assertListEqual(self.model.genome.tolist(), new_model.genome.tolist(), msg="Copying should preserve order")
        new_model.genome[0] = 1
        self.assertEqual(self.model.genome[0], 0, msg="After deep copy original should not change")

    def test_shuffle(self):
        new_model = self.model.copy()
        i = 0
        while np.all(new_model.genome == self.model.genome) and i < 1000:
            new_model.shuffle()
            i += 1
        self.assertFalse(
            np.all(new_model.genome == self.model.genome),
            msg="Shuffling didn't shuffle or you're *really* unlucky"
        )
        self.assertCountEqual(
            self.model.genome.tolist(), new_model.genome.tolist(),
            msg="Shuffling should preserve count"
        )

    def test_get_sites_by_spin(self):
        self.model.genome = np.zeros(len(self.model))
        self.model.genome[1:3] = 1
        self.assertListEqual(self.model.get_sites_by_spin(1).tolist(), [1, 2])

    def test_unique_spins(self):
        self.assertListEqual(
            np.arange(self.model.n_spins).tolist(), self.model.unique_spins.tolist(),
            msg="Should have consecutive spin ids"
        )

    def test_choose(self):
        self.assertIsInstance(self.model.choose(1), Integral, msg="Single choice should be unwrapped")
        self.assertIsInstance(self.model.choose(2), np.ndarray, msg="Multiple choices should come together")
        self.assertEqual(0, self.model.choose(1, mask=self.model.sites == 0), msg="Trivial masking failed")
        self.assertRaises(ValueError, self.model.choose, 999)  # More choices than elements should fail

    def test_genome_controls_structure(self):
        self.assertNotEqual(
            self.model.genome[0], self.model.genome[1],
            msg="This test requires a model with at least two different spins."
        )
        initial_chem = self.model.structure.get_chemical_symbols().copy()
        self.model.genome[[0, 1]] = self.model.genome[[1, 0]]
        self.assertListEqual(
            initial_chem[[0, 1]].tolist(),
            self.model.structure.get_chemical_symbols()[[1, 0]].tolist(),
            msg="Swapping genome should modify the underlying Atoms structure of the model."
        )

    def test_fitness(self):
        self.model.genome = np.random.choice(self.model.unique_spins, len(self.model))
        self.assertEqual(
            len(self.model.fitness_array), len(self.model),
            msg="Fitness array should store one fitness per spin"
        )
        self.assertAlmostEqual(
            self.model.fitness_array.mean(), self.model.fitness,
            msg="System fitness should simply be an average of the individual environments"
        )


class _SpecialTests(ABC):
    """
    unittest.TestCase inheritance implicit, otherwise it tries to actuall test this.
    Class attributes `_class`, `_bad_reps_tuple`, `_ok_reps_tuple`, and `_expected_neighbors` are required.
    """
    @property
    @abstractmethod
    def _class(self):
        pass

    @property
    @abstractmethod
    def _bad_reps_tuple(self):
        pass

    @property
    @abstractmethod
    def _ok_reps_tuple(self):
        pass

    def test_repetitions(self):
        with self.assertRaises(TypeError):
            self._class(repetitions=self._bad_reps_tuple)
        self._class(n_spins=1)
        self._class(repetitions=self._ok_reps_tuple)
        uniform_reps = self._class(repetitions=3)
        self.assertEqual(
            len(uniform_reps._unit_structure) * 3 ** uniform_reps._dimension, len(uniform_reps),
            msg="Integers should repeat along each model dimension."
        )

    def test_neighbors(self):
        model = self._class()
        self.assertTrue(np.all(np.array([len(inds) for inds in model.topology]) == model.n_neighbors))
        with self.assertRaises(ValueError):
            model.n_neighbors = 42


class TestChain1D(TestIsing, _SpecialTests):
    _class = Chain1D
    _bad_reps_tuple = 2, 2
    _ok_reps_tuple = 2
    _expected_neighbors = 2

    def test_check_n_spins_is_valid(self):
        self.assertRaises(ValueError, Chain1D, n_spins=99999)  # Too many spins
        self.assertRaises(ValueError, Chain1D, repetitions=1, n_spins=3)  # Not enough sites

    def test_clean_spin_fractions(self):
        self.assertTrue(np.allclose(self._class._clean_spin_fractions(None, 2), [0.5, 0.5]))
        with self.assertRaises(ValueError):
            Chain1D._clean_spin_fractions([0.25] * 4, 2)  # wrong number
        with self.assertRaises(ValueError):
            Chain1D._clean_spin_fractions([0.5, 0.5, 0.5], 3)  # doesn't sum to 1
        uneven = [0.33333, 0.66667]
        self.assertTrue(np.allclose(Chain1D._clean_spin_fractions(uneven, len(uneven)), uneven))

    def test_set_spins(self):
        self.assertListEqual(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            Chain1D(repetitions=10, n_spins=3, shuffle=False).genome.tolist()
        )

    def test_fitness(self):
        model = Chain1D(repetitions=20, n_spins=2, shuffle=False)
        ideal_fitness = (len(model) - 4) / len(model)
        self.assertEqual(ideal_fitness, model.fitness)
        i = 0
        while model.fitness >= ideal_fitness and i < 1000:
            model.shuffle()
            i += 1
        self.assertLess(
            model.fitness, ideal_fitness,
            msg="Something is wrong in shuffling or fitness, or you got *really* unlucky"
        )

    def test_copy(self):
        model = Chain1D()
        model.genome[0] = 0
        copied_model = model.copy()
        self.assertIsInstance(copied_model, model.__class__)
        self.assertListEqual(model.genome.tolist(), copied_model.genome.tolist(), msg="Genome failed to copy")
        model.genome[0] = 1
        self.assertEqual(0, copied_model.genome[0], msg="Modifying original genome should not effect copy")


class TestSquare2D(TestIsing, _SpecialTests):
    _class = Square2D
    _bad_reps_tuple = 2, 2, 2
    _ok_reps_tuple = 1, 2
    _expected_neighbors = 4


class TestHex2D(TestIsing, _SpecialTests):
    _class = Hex2D
    _bad_reps_tuple = 2, 2, 2
    _ok_reps_tuple = 1, 2
    _expected_neighbors = 6


class TestBCC3D(TestIsing, _SpecialTests):
    _class = BCC3D
    _bad_reps_tuple = 2, 2
    _ok_reps_tuple = 1, 2, 3
    _expected_neighbors = 8


class TestFCC3D(TestIsing, _SpecialTests):
    _class = FCC3D
    _bad_reps_tuple = 2, 2
    _ok_reps_tuple = 1, 2, 3
    _expected_neighbors = 12
