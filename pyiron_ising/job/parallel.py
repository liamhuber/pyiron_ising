# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
Parallelize the Ising approach over multiple random initial conditions.
"""

from pyiron_base import ParallelMaster, JobGenerator
from pyiron_ising.job.ising import Ising
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from functools import cached_property

__author__ = "Liam Huber, Vijay Bhuva"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Oct 20, 2021"


class _IsingGenerator(JobGenerator):
    @property
    def parameter_list(self) -> np.ndarray:
        # Trivial, but without it we get an upstream error about somebody not having a length attribute
        return np.arange(self._master.number_jobs_total)

    @staticmethod
    def modify_job(job: Ising, parameter: None):
        job.model.shuffle()
        job.server.run_mode.non_modal = True
        return job

    def job_name(self, parameter):
        return f"{self._master.job_name}_{parameter}"


class ParallelIsing(ParallelMaster):
    """
    Run multiple Ising calculations on the same model with different random initial conditions.

    Once a template :class:`Ising` instance is provided as a :attribute:`ref_job`, the only other input to specify is
    the :attribute:`server.cores` value, making one child job per core.

    Unlike the :class:`PythonTemplateJob` that :class:`Ising` is based off of, :class:`ParallelMaster` is still quite
    archaic at time of writing, and doesn't have all our more modern data storage handling. To keep life simple, we'll
    sacrifice a little bit of runtime efficiency in exchange for a lot of simplicity and just reload child jobs and
    access their data when we need it. This is all handled under the :attribute:`postprocessing` attribute.
    """
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self._job_generator = _IsingGenerator(self)
        self._post = _PostProcessing(self)

    @property
    def postprocessing(self):
        return self._post

    def validate_ready_to_run(self):
        if self.number_jobs_total is None:
            self.number_jobs_total = self.server.cores
        elif self.number_jobs_total != self.server.cores:
            raise ValueError(
                f"Expected the number of jobs and number of cores to match, but got {self.number_jobs_total} and "
                f"{self.server.cores}, repsectively"
            )

    # And finally, because our inheritance tree sucks, implement and pass the abstract method

    def collect_output(self):
        pass


class _PostProcessing:
    """
    Because I'm willing to sacrifice a little runtime in exchange for saving developer time and never touching HDF
    path strings.
    """
    def __init__(self, parent: ParallelIsing):
        self._p = parent

    @cached_property
    def children(self) -> List[Ising]:
        return [self._p.project.load(name) for name in self._p.child_names]

    @cached_property
    def frame(self) -> np.ndarray:
        child_frame_lengths = [len(c.output.frame) for c in self.children]
        return self.children[np.argmax(child_frame_lengths)].output.frame

    @property
    def end_frame(self) -> np.ndarray:
        return np.array([c.output.frame[-1] for c in self.children])

    @property
    def median_end_frame(self) -> float:
        return np.median(self.end_frame)

    @cached_property
    def padded_child_fitness(self) -> np.ndarray:
        return np.array([
            np.pad(c.output.fitness, pad_width=(0, len(self.frame) - len(c.output.fitness)), mode='edge')
            for c in self.children
        ])

    @property
    def median_fitness(self) -> np.ndarray:
        return np.median(self.padded_child_fitness, axis=0)

    @property
    def std_fitness(self) -> np.ndarray:
        return np.std(self.padded_child_fitness, axis=0)

    def plot3d(self, child=0, frame=-1, **kwargs):
        return self.children[child].plot3d(frame=frame, **kwargs)

    def plot(
            self, ax=None,
            label_x=True, label_y=True,
            show_success_log=False, show_neutral_log=False,
            show_legend=True,
            show_solution=True,
            logx=True, logy=False,
            child_alpha=0.5, child_style='--',
            **plot_kwargs
    ):
        """
        Show the median fitness as a function of step.

        Error bars give roughly the 95% confidence interval, i.e. 1.96 times the standard deviation among child
        fitnesses at each frame. (We're mixing mean and median here and not making an special consideration for small
        populations, but it's still qualitatively useful to see.)
        """
        if ax is None:
            _, ax = plt.subplots()

        for c in self.children:
            c.plot(
                ax=ax,
                label_x=label_x, label_y=label_y, label_line=False,
                show_success_log=show_success_log, show_neutral_log=show_neutral_log,
                show_legend=False,
                logx=logx, logy=logy,
                alpha=child_alpha,
                linestyle=child_style
            )

        ax.errorbar(
            self.frame,
            self.median_fitness,
            1.96 * self.std_fitness,
            label='median fitness',
            **plot_kwargs
        )

        if show_solution:
            ax.axvspan(self.end_frame.min(), self.end_frame.max(), alpha=0.5 * child_alpha)
            ax.axvline(self.median_end_frame, color='k', linestyle='--', label='median steps to solition')
            ax.axvspan(
                self.median_end_frame - self.end_frame.std(),
                self.median_end_frame + self.end_frame.std(),
                alpha=0.5 * child_alpha
            )

        if show_legend:
            ax.legend()

        return ax
