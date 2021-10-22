# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from ._ising import TestIsing
from pyiron_ising.mutate import Flip, Swap, Cluster, Mutator
from pyiron_ising.toolkit import ModelFactory
import numpy as np


class TestMutations(TestIsing):
    def setUp(self):
        super().setUp()
        self.before = self.model.genome.copy()

    def test_flip(self):
        identifier, site = Flip()(self.model)
        self.assertEqual("flip", identifier)
        self.assertNotEqual(self.model.genome[site], self.before[site])

    def test_swap(self):
        identifier, i, j = Swap()(self.model)
        self.assertEqual("swap", identifier)
        self.assertNotEqual(i, j)
        self.assertNotEqual(self.model.genome[i], self.model.genome[j])
        self.assertListEqual(self.model.genome[[i, j]].tolist(), self.before[[j, i]].tolist())

    def test_swap_naive(self):
        self.assertGreaterEqual(len(self.model), 3, msg='The test model needs at lest three sites for this test')
        self.model.genome[:-1] = 0
        self.model.genome[-1] = 1
        naive = Swap(naive=True)
        n = 0
        i, j = 0, -1
        while self.model.genome[i] != self.model.genome[j] and n < 10000:
            _, i, j = naive(self.model)
            self.assertNotEqual(i, j)
        self.assertEqual(
            self.model.genome[i], self.model.genome[j],
            msg="Naive should allow swapping similar sites."
        )

    def test_cluster_plain(self):
        """Just gets all the sites of the same type"""
        model = ModelFactory().FCC3D(repetitions=3, shuffle=False)
        before = model.genome.copy()
        identifier, _, _ = Cluster()(model)
        self.assertFalse(
            np.any(before == model),
            msg="Expected all values of same species to be found and genome to get perfectly flipped"
        )

    def test_cluster_xenophobic(self):
        """Likes the internal columns with only similar neighbors"""
        model = ModelFactory().Square2D(repetitions=(8, 1), shuffle=False)
        og_genome = model.genome.copy()
        mutation = Cluster(min_like_neighbors=model.n_neighbors)

        identifier, i, j = mutation(model, 0, 4)
        self.assertEqual(identifier, "swap", msg="Initial sites don't meet criterion, so should revert to pair swap")
        self.assertListEqual(
            [1, 0, 0, 0, 0, 1, 1, 1], model.genome.tolist(),
            msg="Initial sites don't meet criterion, so should have been returned alone"
        )
        model.genome = og_genome.copy()  # Reset genome

        identifier, i, j = mutation(model, 1, 5)
        self.assertEqual(identifier, "cluster_min4")
        self.assertListEqual(
            [0, 1, 1, 0, 1, 0, 0, 1], model.genome.tolist(),
            msg="Should have found interior columns in both cases"
        )
        model.genome = og_genome.copy()  # Reset genome

        identifier, i, j = mutation(model, 0, 5)
        self.assertEqual(identifier, "swap", msg="Failure of one cluster should lead to early truncation")
        self.assertListEqual(
            [1, 0, 0, 0, 1, 0, 1, 1], model.genome.tolist(),
            msg="Failure of one cluster should lead to early truncation"
        )

    def test_cluster_xenophilic(self):
        """Likes the stripes neighbouring the other phase"""
        model = ModelFactory().Hex2D(repetitions=(4, 4))
        model.genome[:] = 0
        model.genome[model.structure.positions[:, 1] >= 0.5 * model.structure.cell[1, 1]] = 1
        # [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        og_genome = model.genome.copy()
        mutation = Cluster(max_like_neighbors=4)

        identifier, i, j = mutation(model, 1, 5)
        self.assertEqual(identifier, "swap", msg="Initial sites don't meet criterion, so should revert to pair swap")
        self.assertListEqual(
            [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            model.genome.tolist(),
            msg="Initial sites don't meet criterion, so should have been returned alone"
        )
        model.genome = og_genome.copy()  # Reset genome

        identifier, i, j = mutation(model, 0, 4)
        self.assertEqual(identifier, "cluster_max4")
        self.assertListEqual(
            np.roll(
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], 1
            ).tolist(),
            model.genome.tolist(),
            msg="Should have edge stripes in both cases"
        )
        model.genome = og_genome.copy()  # Reset genome

        identifier, i, j = mutation(model, 0, 5)
        self.assertEqual(identifier, "swap", msg="Failure of one cluster should lead to early truncation")
        self.assertListEqual(
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            model.genome.tolist(),
            msg="Failure of one cluster should lead to early truncation"
        )

    def test_cluster_neighbor_combos(self):
        model = ModelFactory().Square2D(repetitions=(8, 1), shuffle=False)
        before = model.genome.copy()

        identifier, _, _ = Cluster(min_like_neighbors=0, max_like_neighbors=model.n_neighbors)(model)
        self.assertEqual("cluster_min0_max4", identifier)
        self.assertFalse(np.any(model.genome == before), msg="Trivial conditions should trivially give full inversion.")
        model.genome = before.copy()

        identifier, i, j = Cluster(min_like_neighbors=model.n_neighbors + 1, max_like_neighbors=0)(model)
        self.assertEqual("swap", identifier)
        self.assertListEqual(
            before[[i, j]].tolist(), model.genome[[j, i]].tolist(),
            msg="Impossible conditions should give swap."
        )


class TestMutator(TestIsing):
    def setUp(self):
        super().setUp()
        self.m = Mutator()

    def test_adding(self):
        self.m.add.Flip()
        self.m.add.Swap()
        self.m.add.Swap(naive=True)
        self.m.add.Cluster()
        self.assertEqual(4, len(self.m.mutations))
        self.assertEqual(4, len(self.m))
        self.assertFalse(self.m.mutations[1].naive)
        self.assertTrue(self.m.mutations[2].naive)

    def test_selecting(self):
        # Do a stats weighted check after repeated calls
        self.m.add.Flip(2)
        self.m.add.Swap(1, naive=True)
        flipped = [self.m(self.model)[0] == 'flip' for _ in np.arange(100000)]
        flip_frac = np.sum(flipped) / len(flipped)
        self.assertAlmostEqual(
            self.m.mutations[0].weight / (self.m.mutations[0].weight + self.m.mutations[1].weight),
            flip_frac,
            places=2,
            msg="Either the mutator is not choosing mutations correctly, or you got really unlucky."
        )
