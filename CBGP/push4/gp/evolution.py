from abc import abstractmethod, ABC
from typing import Callable, Tuple

import numpy as np
import time

from push4.gp.individual import Individual
from push4.gp.population import Population
from push4.gp.selection import Selector
from push4.gp.spawn import Spawner
from push4.gp.variation import VariationOperator
from push4.lang.dag import Dag
from push4.utils import escape


def _spawn_individual(spawner, genome_size, output_type: type, *args):
    return Individual(spawner.spawn_genome(*genome_size), output_type)


class Evolver(ABC):

    def __init__(self,
                 error_function: Callable[[Dag], np.array],
                 spawner: Spawner,
                 selector: Selector,
                 variation: VariationOperator,
                 population_size: int,
                 max_generations: int,
                 initial_genome_size: Tuple[int, int],
                 downsample_rate: float = 1.,
                 alpha: float = 1.):
        self.error_function = error_function
        self.spawner = spawner
        self.selector = selector
        self.variation = variation
        self.population_size = population_size
        self.max_generations = int(max_generations/downsample_rate)
        self.initial_genome_size = initial_genome_size
        self.population = None
        self.generation = 0
        self.best_seen = None
        self.downsample_rate = downsample_rate
        self.alpha = alpha  # lexiprob
        self.logs = {
            'lexiprob_runtime': [],
            'lexiprob_comparisons': [],
            'lexicase_runtime': [],
            'unique_overlap': [],
            'total_overlap': [],
            'eval_runtime': [],
            'produce_runtime': [],
            'total_runtime': [],
            'downsample_rate': downsample_rate,
            'alpha': alpha,
            'train_err': []
        }

        # TODO: Add ParallelContext

    def init_population(self, output_type: type):
        """Initialize the population."""
        self.population = Population()
        for i in range(self.population_size):
            self.population.add(_spawn_individual(
                self.spawner, self.initial_genome_size, output_type))

    @abstractmethod
    def step(self, output_type: type):
        """Perform one generation (step) of evolution. Return if should continue.
        The step method should assume an evaluated Population, and must only
        perform parent selection and variation (producing children). The step
        method should modify the search algorithms population in-place, or
        assign a new Population to the population attribute.
        """
        pass

    def _full_step(self, output_type) -> bool:
        t = time.time()
        self.generation += 1
        self.population.evaluate(
            self.error_function, self.downsample_rate, self.alpha)
        self.logs['eval_runtime'].append(time.time()-t)

        best_this_gen = self.population.best()
        if self.best_seen is None or best_this_gen.total_error < self.best_seen.total_error:
            self.best_seen = best_this_gen

        best_is_valid = self.best_seen.program is not None
        print("{gn}\t\t{me}\t\t{be}\t\t{dv}\t\t{best_err}\t\t{best_code}".format(
            gn=round(self.generation, 3),
            me=round(self.population.median_error(), 3),
            be=round(self.population.best().total_error, 3),
            dv=round(self.population.error_diversity(), 3),
            best_err=self.best_seen.total_error,
            best_code=escape(self.best_seen.program.root.to_code()
                             ) if best_is_valid else "NA"
        ))
        # self.best_seen.program.pprint()

        self.logs['train_err'].append(float(best_this_gen.total_error))

        if self._is_solved():
            return False

        self.step(output_type)
        self.logs['total_runtime'].append(time.time()-t)
        return True

    def _is_solved(self):
        return round(self.best_seen.total_error, 6) == 0

    def run(self, output_type: type) -> Individual:
        """Run the algorithm until termination."""
        self.init_population(output_type)

        print("Gen\t\tMedian\t\tMAD\t\tBest\t\tDiv\t\tRun Best\t\tCode")
        while self._full_step(output_type):
            if self.generation >= self.max_generations:
                break

        if self._is_solved():
            print("Solution found.")
        else:
            print("No solution found.")

        return self.best_seen


class GeneticAlgorithm(Evolver):
    """Genetic algorithm to synthesize Push programs.
    An initial Population of random Individuals is created. Each generation
    begins by evaluating all Individuals in the population. Then the current
    Population is replaced with children produced by selecting parents from
    the Population and applying VariationOperators to them.
    """

    def _make_child(self, output_type: type) -> Individual:
        parent_genomes = [p.genome for p in self.selector.select(
            self.population, n=self.variation.num_parents)]
        child_genome = self.variation.produce(parent_genomes, self.spawner)
        return Individual(child_genome, output_type)

    def step(self, output_type: type):
        """Perform one generation (step) of the genetic algorithm.
        The step method assumes an evaluated Population and performs parent
        selection and variation (producing children).
        """

        new_population = Population(
            [self._make_child(output_type)
             for _ in range(self.population_size)]
        )
        for item in self.population.logs:
            self.logs[item].append(self.population.logs[item])
        self.population = new_population


class Lexiprob(Evolver):
    """Genetic algorithm to synthesize Push programs.
    An initial Population of random Individuals is created. Each generation
    begins by evaluating all Individuals in the population. Then the current
    Population is replaced with children produced by selecting parents from
    the Population and applying VariationOperators to them.
    """

    def _make_child(self, output_type: type) -> Individual:
        # candidates - list of lists of candidates with same error vector
        candidates, p = self.population.nondom_set
        parent_idx = np.random.choice(
            np.arange(len(candidates)), p=p, replace=True, size=self.variation.num_parents
        )
        parents = [np.random.choice(
            candidates[i]) for i in parent_idx]
        parent_genomes = [p.genome for p in parents]
        parent_hashes = [p._error_vector_bytes for p in parents]
        child_genome = self.variation.produce(parent_genomes, self.spawner)
        return Individual(child_genome, output_type), parent_hashes

    def _make_child_lexicase(self, output_type: type) -> Individual:
        parents = [p for p in self.selector.select(
            self.population, n=self.variation.num_parents)]
        parent_genomes = [p.genome for p in parents]
        parent_hashes = [p._error_vector_bytes for p in parents]
        child_genome = self.variation.produce(parent_genomes, self.spawner)
        return Individual(child_genome, output_type), parent_hashes

    def step(self, output_type: type):
        """Perform one generation (step) of the genetic algorithm.
        The step method assumes an evaluated Population and performs parent
        selection and variation (producing children).
        """
        lexicase_parents = []
        for _ in range(self.population_size):
            lexicase_parents += self._make_child_lexicase(output_type)[1]

        t = time.time()
        lexiprob_children = []
        lexiprob_parents = []
        for _ in range(self.population_size):
            child, parents = self._make_child(output_type)
            lexiprob_children.append(child)
            lexiprob_parents += parents
        self.logs['produce_runtime'].append(time.time()-t)

        I = 0
        T = 0
        for p in set(lexicase_parents):
            if p in set(lexiprob_parents):
                I += 1
            T += 1
        print('unique overlap: {}/{}={}'.format(I, T, I/T))
        self.logs['unique_overlap'].append((I, T))

        lexicase_count = {}
        for key in set(lexicase_parents):
            lexicase_count[key] = 0
        for p in lexicase_parents:
            lexicase_count[p] += 1

        lexiprob_count = {}
        for key in set(lexiprob_parents):
            lexiprob_count[key] = 0
        for p in lexiprob_parents:
            lexiprob_count[p] += 1

        I = 0
        T = 0
        for key in lexicase_count:
            if key in lexiprob_count:
                I += min(lexicase_count[key], lexiprob_count[key])
            T += lexicase_count[key]

        print('total overlap: {}/{}={}'.format(I, T, I/T))
        self.logs['total_overlap'].append((I, T))

        for item in self.population.logs:
            self.logs[item].append(self.population.logs[item])

        self.population = Population(lexiprob_children)
