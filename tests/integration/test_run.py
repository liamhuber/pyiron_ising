# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import TestWithCleanProject
import pyiron_ising
import numpy as np


class TestRun(TestWithCleanProject):
    def test_phase_separation(self):
        job = self.project.ising.job.Ising('mc', delete_existing_job=True)
        job.input.model = self.project.ising.model.Square2D(repetitions=8, n_spins=4)
        job.input.mutations.add.Swap()
        job.input.mutations.add.Cluster(max_like_neighbors=2)
        job.input.mutations.add.Cluster(min_like_neighbors=2)
        job.input.n_steps = 500
        job.input.stopping_fitness = 0.1

        parallel = self.project.ising.job.ParallelIsing('para', delete_existing_job=True)
        parallel.ref_job = job
        parallel.server.cores = 4

        parallel.run()
        post_fitness = np.array(parallel.postprocessing.median_fitness)
        self.assertGreater(post_fitness[-1], post_fitness[0], msg="Fitness did not improve")
        self.assertLess(
            parallel.postprocessing.std_fitness[-1], parallel.postprocessing.std_fitness[0],
            msg="With such a low stopping fitness, we expect most jobs to hit this and thus reduce variance"
        )

        loaded = self.project.load('para')
        self.assertTrue(np.allclose(post_fitness, loaded.postprocessing.median_fitness))

    def test_bfs_ordering(self):
        """
        Double BFS was broken because numpy's unique call didn't preserve order, so when the larger of mismatched
        clusters got truncate, the truncation success depended on the actual id values. I ran into it by chance, so I'm
        just going to use the exact case I stumbled upon that caused it, because carefully constructing a case where BFS struggles
        due to index order sounds like a huge pain!
        """
        hex_ = self.project.ising.model.Hex2D(repetitions=(8, 5))
        og_genome = np.array([
            0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
            0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
            1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1
        ])
        hex_.genome = og_genome.copy()
        mutation = self.project.ising.mutation.Cluster(max_like_neighbors=3)
        mutation(hex_, 17, 18)
        changes = hex_.genome != og_genome
        group_01 = np.argwhere(changes * (og_genome == 0)).T[0]
        group_10 = np.argwhere(changes * (og_genome == 1)).T[0]
        self.assertEqual(
            set(group_01), set([18, 19, 8, 29, 77]),
            msg="This is the bigger cluster and 76 and 77 should have gotten truncated."
        )
        self.assertEqual(
            set(group_10), set([17, 7, 28, 6, 75]),
            msg="This is the smaller cluster and should be the easier one to get right."
        )

