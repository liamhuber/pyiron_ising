"""
Some helper classes for running jobs with higher throughput, slapped together post-facto to make life easier during plotting
"""

from pyiron_ising import Project, Model, Mutation, Chain1D, Square2D, Hex2D, BCC3D, Swap, Cluster
from typing import Type, List, Callable, Union
from functools import cached_property
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

QUEUE = 'cm'  # Remote queue to run on, set to None to use your local machine
CORES = 40  # How many cores to use, i.e. independent runs for each tests to get statistics
RUNTIME = 4*24*60*60 - 1  # 4 days in seconds, only used if QUEUE is not None
MAXSTEPS = int(1E6)  # The max number of iterations to use

class Scaling(ABC):
    def __init__(
            self,
            project: Project,
            name: str,
            reps: List[int],
            min_like_neighbors=None,
            max_like_neighbors=None,
            n_steps: int = MAXSTEPS,
            n_print: int = MAXSTEPS,
            log_mutations: bool = False,
            cores: int = CORES if CORES is not None else 1,
            queue: Union[str, None] = QUEUE,
            run_time: Union[int, None] = RUNTIME,
            delete_existing_job: bool = False
    ):
        self.project = project
        self.name = name
        self.min_like_neighbors = min_like_neighbors
        self.max_like_neighbors = max_like_neighbors
        self._mutations = None

        self.jobs = []
        size = []
        for r in reps:
            ref = project.ising.job.Ising(f'{name}_ref_{r}', delete_existing_job=delete_existing_job)
            ref.input.model = self.model(repetitions=r, n_spins=2)
            for m in self.mutations:
                ref.input.mutations.add(m)
            ref.input.stopping_fitness = self.optimal_fitness_function(r)
            ref.input.n_steps = n_steps
            ref.input.n_print = n_print
            ref.input.log_mutations = log_mutations

            job = project.ising.job.ParallelIsing(f'{name}_{r}', delete_existing_job=delete_existing_job)
            job.ref_job = ref
            job.server.cores = cores
            if queue is not None:
                job.server.queue = queue
                if run_time is not None:
                    job.server.run_time = run_time
            self.jobs.append(job)
            size.append(len(ref.input.model))

        self.size = np.array(size)
        self.reps = np.array(reps)

    @property
    @abstractmethod
    def model(self) -> Type[Model]:
        pass

    @property
    @abstractmethod
    def mutations(self) -> List[Mutation]:
        pass

    @abstractmethod
    def optimal_fitness_function(self) -> Callable:
        pass

    def run(self) -> None:
        for j in self.jobs:
            j.run()

    # @property
    # def finished_jobs(self):
    #     return [j for j in self.jobs if j.status == 'finished']

    @cached_property
    def name_mask(self):
        return np.any(
            [self.project.job_table().job.values == j.name for j in self.jobs],
            axis=0
        )

    @cached_property
    def run_time(self):
        """Actual compute time for the job"""
        return self.project.job_table().totalcputime.values[self.name_mask]

    @cached_property
    def run_steps(self):
        """Median steps (to either solution or step limit)"""
        return np.array([job.postprocessing.median_end_frame for job in self.jobs])

    @cached_property
    def run_confidence(self):
        """95% confidence interval for the *mean* steps (assuming they're gaussian anyhow)"""
        return np.array([2.96 * job.postprocessing.end_frame.std() for job in self.jobs])

    @cached_property
    def run_bounds(self):
        """Min and max steps to solution"""
        mins = [job.postprocessing.end_frame.min() for job in self.jobs]
        maxs = [job.postprocessing.end_frame.max() for job in self.jobs]
        return np.array([mins, maxs])

    def _figax(self, ax):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        return fig, ax

    def plot(
            self,
            ax=None,
            xlabel="System size",
            steps_ylabel="Median steps to solution",
            show_time=False,
            time_ylabel="Run time",
            label='steps',
            marker='x',
            linestyle='none',
            fname=None,
            fmt='eps',
            use_error=True,
            x_style='sci',
            y_style='sci',
            **errorbar_kwargs

    ):
        fig, ax = self._figax(ax)

        ax.errorbar(
            self.size.astype(float),
            self.run_steps,
            yerr=self.run_bounds if use_error else None,
            linestyle=linestyle,
            marker=marker,
            label=label,
            **errorbar_kwargs
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(steps_ylabel)
        ax.ticklabel_format(axis='x', style=x_style)
        ax.ticklabel_format(axis='y', style=y_style)

        if show_time:
            time_ax = ax.twinx()
            time_ax.scatter(self.size, self.run_time, color='r', marker='+', label='time')
            time_ax.set_ylabel(time_ylabel)

        if fname is not None:
            plt.tight_layout()
            plt.savefig(fname=f'{fname}.{fmt}', fmt=fmt)

        return fig, ax

    def plot_normalized(
            self,
            ax=None,
            xlabel="System size", ylabel="Normalized median steps",
            function=None,
            show_legend=True,
            fname=None,
            fmt='eps',
            x_style='plain',
            y_style='sci',
            **scatter_kwargs
    ):
        fig, ax = self._figax(ax)
        function = function if function is not None else lambda x: x

        ax.scatter(
            self.size,
            self.run_steps / function(self.size),
            **scatter_kwargs
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='x', style=x_style)
        ax.ticklabel_format(axis='y', style=y_style)

        if show_legend:
            ax.legend()

        if fname is not None:
            plt.tight_layout()
            plt.savefig(fname=f'{fname}.{fmt}', fmt=fmt)

        return fig, ax


def _optimal_score(n_tot, n_border, border_score):
    n_interior = n_tot - n_border
    return (n_interior + border_score * n_border) / n_tot


class Chain(ABC):
    @property
    def model(self):
        return Chain1D

    @staticmethod
    def optimal_fitness_function(reps):
        return _optimal_score(reps, 4, (1 - 1) / 2)


class Square(ABC):
    @property
    def model(self):
        return Square2D

    @staticmethod
    def optimal_fitness_function(reps):
        return _optimal_score(reps ** 2, 4 * reps, ((3 - 1) / 4))


class Hex(ABC):
    @property
    def model(self):
        return Hex2D

    @staticmethod
    def optimal_fitness_function(reps):
        return _optimal_score(2 * reps ** 2, 4 * reps, ((4 - 2) / 6))


class BCC(ABC):
    @property
    def model(self):
        return BCC3D

    @staticmethod
    def optimal_fitness_function(reps):
        return _optimal_score(2 * reps ** 3, 4 * reps ** 2, (4 - 4) / 8)


class Swapper(ABC):
    @property
    def mutations(self):
        if self._mutations is None:
            self._mutations = [Swap()]
        return self._mutations


class Clusterer(ABC):
    @property
    def mutations(self):
        if self._mutations is None:
            self._mutations = [
                Cluster(min_like_neighbors=self.min_like_neighbors),
                Cluster(max_like_neighbors=self.max_like_neighbors)
            ]
        return self._mutations


class ChainSwap(Swapper, Chain, Scaling):
    pass


class ChainCluster(Clusterer, Chain, Scaling):
    pass


class SquareSwap(Swapper, Square, Scaling):
    pass


class SquareCluster(Clusterer, Square, Scaling):
    pass


class HexSwap(Swapper, Hex, Scaling):
    pass


class HexCluster(Clusterer, Hex, Scaling):
    pass


class BCCSwap(Swapper, BCC, Scaling):
    pass


class BCCCluster(Clusterer, BCC, Scaling):
    pass


class Experiments:
    def __init__(self, project):
        self.project = project
        
    def job_table(self, unfinished_only=False):
        mask = self.project.job_table().hamilton.values == 'ParallelIsing'
        if unfinished_only:
            mask *= self.project.job_table().status.values != 'finished'
        return self.project.job_table()[mask]
    
    @cached_property 
    def chain_swap(self):
        return ChainSwap(
            self.project, 
            'chain_swap', 
            [16, 32, 48, 64, 80, 96, 112, 128],
        )
    
    @cached_property
    def chain_cluster(self):
        return ChainCluster(
            self.project, 
            'chain_cluster', 
            [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        )
    
    @cached_property
    def square_swap(self):
        return SquareCluster(
            self.project, 
            'squre_swap',
            [4, 6, 8, 10, 12],
        )

    @cached_property
    def square_clusters(self):
        return [
            SquareCluster(
                self.project, 
                f'square_cluster{i}',
                [4, 8, 12, 16, 24, 28, 32,],  # 48, 64, 96, 128],  # Died by 48
                min_like_neighbors=i,
                max_like_neighbors=i,
            )
            for i in [2, 3]
        ]
    
    @cached_property
    def hex_swap(self):
        return HexSwap(
            self.project, 
            'hex_swap',
            [4, 6, 8, 10, 12],
        )

    @cached_property
    def hex_clusters(self):
        return [
            HexCluster(
                self.project, 
                f'hex_cluster{i}',
                [4, 6, 8, 10, 12, 14, 16,],  # 32, 48, 64, 96, 128],  # Died by 32
                min_like_neighbors=i,
                max_like_neighbors=i,
            )
            for i in [2, 3, 4, 5]
        ]
    
    @cached_property
    def bcc_swap(self):
        return BCCSwap(
            self.project, 
            'bcc_swap',
            [2, 3, 4, 5],
        )
    
    @cached_property
    def bcc_clusters(self):
        return [
            BCCCluster(
                self.project, 
                f'bcc_cluster{i}',
                [2, 3, 4, 5, 6, 7, 8,],  # 16, 24, 32],  # Timed out at 16
                min_like_neighbors=i,
                max_like_neighbors=i,
            )
            for i in [3, 4, 5, 6]
        ]
