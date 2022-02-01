# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""
Evolve an Ising model.
"""

from pyiron_base import PythonTemplateJob, DataContainer
from pyiron_ising.model import Model
from pyiron_ising.mutate import Mutator
from typing import Type, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

__author__ = "Liam Huber, Vijay Bhuva"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "production"
__date__ = "Oct 18, 2021"


class IsingInput(DataContainer):
    def __init__(self, init=None, table_name='input', lazy=False):
        super().__init__(init=init, table_name=table_name, lazy=lazy)
        self._model = None
        self._mutations = Mutator()
        self.n_steps = 1000
        self.n_print = 10
        self.log_mutations = True
        self.stopping_fitness = None
        # self.temperature = 0

    @property
    def model(self) -> Type[Model]:
        return self._model

    @model.setter
    def model(self, m: Type[Model]):
        if not isinstance(m, Model):
            raise TypeError(f'Expected a Model but got {type(m)}')
        self._model = m

    @property
    def mutations(self) -> Mutator:
        return self._mutations


class IsingOutput(DataContainer):
    def __init__(self, init=None, table_name='input', lazy=False):
        super().__init__(init=init, table_name=table_name, lazy=lazy)
        self._mutation_log = []
        self._frame = []
        self._genome = []
        self._fitness = []

    def log_mutation(self, step, mutation_info, fitness_change):
        # TODO: ?Make a mutation info data object?
        self._mutation_log.append({
            'step': step,
            'mutation': mutation_info[0],
            'seeds': mutation_info[1:],
            'dfitness': fitness_change
        })

    def collect_frame(self, n: int, model: Type[Model]):
        self._frame.append(n)
        self._genome.append(model.genome.copy())
        self._fitness.append(model.fitness)

    @property
    def mutation_log(self) -> dict:
        return self._mutation_log

    @property
    def frame(self) -> List[int]:
        return self._frame

    @property
    def genome(self) -> List[List[int]]:
        return self._genome

    @property
    def fitness(self) -> List[float]:
        return self._fitness


class Ising(PythonTemplateJob):
    """
    Run an Ising calculation.

    Input:
        model (Model): The model to evolve.
        mutations (Mutator): The mutations to evolve it. (Defined by default, so you can directly call
            `input.mutations.add.` to add mutations.
        n_steps (int): The maximum number of iterations to run. (Default is 1000.)
        n_print (int): The interval between saving the genome and fitness as we run. (Default is 10.)
        log_mutations (bool): Whether to log each accepted mutation. (Default is True)
        stopping_fitness (float): An early-stopping criterion to kill the job once the fitness reaches or surpasses
            this value. (Default is None, just run all steps.)

    Output:
        frame (list[int]): The step at which we save data.
        genome (list[list[int]]): The genome for each frame.
        fitness (list[float]): The fitness for each frame.
        mutation_log (dict): A dictionary logging each successful or neutral mutation. Entries are 'step'
            (obvious meaning, but note that it's not bound by the printing interval like other data), 'mutation' (a
            string identifier for which mutation it was), 'seeds' (any integer site ids which were used to generate
            the specific mutation instance), and 'dfitness' (how much the fitness changed on mutation.
    """
    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self.storage.input = IsingInput(init=self.storage.input)
        self.storage.output = IsingOutput(init=self.storage.output)

    @property
    def mutate(self) -> Mutator:
        """Shortcut for invoking the call method of the mutator"""
        return self.input.mutations

    @property
    def model(self) -> Type[Model]:
        return self.input.model

    def validate_ready_to_run(self):
        if len(self.input.mutations.normalized_weights) < 1:
            raise ValueError("Insufficient mutations found, please use input.mutations.add")
        if self.input.model is None:
            raise TypeError("No model found, please set input.model")

    def run_static(self):
        self.status.running = True
        self.output.collect_frame(0, self.model)

        for n in np.arange(self.input.n_steps):
            old_fitness = self.model.fitness
            old_genome = self.model.genome.copy()
            mutation_info = self.mutate(self.model)
            fitness = self.model.fitness
            if fitness < old_fitness:
                self.model.genome = old_genome  # Revert
            elif self.input.log_mutations:
                self.output.log_mutation(n, mutation_info, fitness - old_fitness)

            if n % self.input.n_print == 0:
                self.output.collect_frame(n + 1, self.model)

            if self.input.stopping_fitness is not None and fitness >= self.input.stopping_fitness:
                break

        if n % self.input.n_print != 0:
            self.output.collect_frame(n + 1, self.model)

        self.status.finished = True
        self.to_hdf()

    def plot3d(self, frame=-1, **kwargs):
        model = self.model.copy()
        model.genome = self.output.genome[frame]
        return model.plot3d(**kwargs)

    def plot(
            self, ax=None,
            label_x=True, label_y=True, label_line=True,
            show_success_log=True, show_neutral_log=False,
            show_legend=True,
            logx=False, logy=False,
            **plot_kwargs
    ):
        """Show the fitness as a function of step."""
        if ax is None:
            _, ax = plt.subplots()

        if show_success_log or show_neutral_log:
            self._plot_log(ax, show_success_log, show_neutral_log, label_line)

        if label_line:
            plot_kwargs['label'] = 'fitness'
        ax.plot(self.output.frame, self.output.fitness, **plot_kwargs)

        if self.input.stopping_fitness is not None:
            ax.axhline(self.input.stopping_fitness, linestyle=':', color='k')

        if label_x:
            ax.set_xlabel("Step")

        if label_y:
            ax.set_ylabel("Fitness")

        if (show_success_log or show_neutral_log) and show_legend:
            ax.legend()

        if logx:
            ax.set_xscale('log')

        if logy:
            ax.set_yscale('log')

        return ax

    def _plot_log(self, ax, show_success_log, show_neutral_log, label_line):
        """Add barcodes for neutral and beneficial mutations"""
        mutation_identifiers = np.unique([d['mutation'] for d in self.output.mutation_log])
        colormap = {m: c for m, c in zip(
            mutation_identifiers,
            sns.color_palette(n_colors=len(mutation_identifiers))
        )}
        labeled = []
        for d in self.output.mutation_log:
            neutral = np.isclose(d['dfitness'], 0)
            if not show_neutral_log and neutral:
                continue
            if not show_success_log and not neutral:
                continue
            ax.axvline(
                d['step'] + 0.5,
                color=colormap[d['mutation']],
                linestyle='--' if neutral else '-',
                label=None if neutral or d['mutation'] in labeled else d['mutation'] or not label_line
            )
            if not neutral:
                labeled.append(d['mutation'])
        return ax

    # And finally, because our inheritance tree sucks, implement and pass all the abstract methods

    def write_input(self):
        pass

    def collect_output(self):
        pass

    def run_if_interactive(self):
        pass

    def interactive_close(self):
        pass

    def interactive_fetch(self):
        pass

    def interactive_flush(self, path="generic", include_last_step=True):
        pass

    def run_if_interactive_non_modal(self):
        pass

    def run_if_refresh(self):
        pass

    def _run_if_busy(self):
        pass
