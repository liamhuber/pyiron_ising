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
