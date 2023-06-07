"""The :mod:`selection` module defines classes to select Individuals from Populations."""
from abc import ABC, abstractmethod
from copy import copy
from typing import Sequence, Union
from operator import attrgetter

import numpy as np
from numpy.random import random, choice, shuffle

from push4.gp.individual import Individual
from push4.gp.population import Population
from push4.utils import median_absolute_deviation
import time


class Selector(ABC):
    """Base class for all selection algorithms."""

    @abstractmethod
    def select_one(self, population: Population) -> Individual:
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        pass

    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
        """Return `n` individuals from the population.

        Parameters
        ----------
        population : Population
            A Population of Individuals.
        n : int
            The number of parents to select from the population. Default is 1.

        Returns
        -------
        Sequence[Individual]
            The selected Individuals.

        """
        selected = []
        for i in range(n):
            selected.append(self.select_one(population))
        return selected


class FitnessProportionate(Selector):
    """Fitness proportionate selection, also known as roulette wheel selection.

    See: https://en.wikipedia.org/wiki/Fitness_proportionate_selection
    """

    def select_one(self, population: Population) -> Individual:
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        return self.select(population)[0]

    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
        """Return `n` individuals from the population.

        Parameters
        ----------
        population
            A Population of Individuals.
        n : int
            The number of parents to select from the population. Default is 1.

        Returns
        -------
        Sequence[Individual]
            The selected Individuals.

        """
        population_total_errors = np.array([i.total_error for i in population])
        sum_of_total_errors = np.sum(population_total_errors)
        probabilities = 1.0 - (population_total_errors / sum_of_total_errors)
        selected_ndxs = np.searchsorted(np.cumsum(probabilities), random(n))
        return [population[ndx] for ndx in selected_ndxs]


class Tournament(Selector):
    """Tournament selection.

    See: https://en.wikipedia.org/wiki/Tournament_selection

    Parameters
    ----------
    tournament_size : int, optional
        Number of individuals selected uniformly randomly to participate in
        the tournament. Default is 7.

    Attributes
    ----------
    tournament_size : int, optional
        Number of individuals selected uniformly randomly to participate in
        the tournament. Default is 7.

    """

    def __init__(self, tournament_size: int = 7):
        self.tournament_size = tournament_size

    def select_one(self, population: Population) -> Individual:
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        tournament = choice(population, self.tournament_size, replace=False)
        return min(tournament, key=attrgetter('total_error'))


class CaseStream:

    def __init__(self, n_cases: int):
        self.cases = list(range(n_cases))

    def __iter__(self):
        shuffle(self.cases)
        for case in self.cases:
            yield case


def one_individual_per_error_vector(population: Population) -> Sequence[Individual]:
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which indiviudal gets selected.
    """
    population_list = list(copy(population))
    shuffle(population_list)
    preselected = []
    error_vector_hashes = set()
    for individual in population_list:
        error_vector_hash = hash(individual.error_vector_bytes)
        if error_vector_hash not in error_vector_hashes:
            preselected.append(individual)
            error_vector_hashes.add(error_vector_hash)
    return preselected


class Lexicase(Selector):
    """Lexicase Selection.

    All training cases are considered iteratively in a random order. For each
    training cases, the population is filtered to only contain the Individuals
    which have an error value within epsilon of the best error value on that case.
    This filtering is repeated until the population is down to a single Individual
    or all cases have been used. After the filtering iterations, a random
    Individual from the remaining set is returned as the selected Individual.

    See: https://ieeexplore.ieee.org/document/6920034
    """

    def __init__(self, epsilon: Union[bool, float, np.ndarray] = False):
        self.epsilon = epsilon

    @staticmethod
    def _epsilon_from_mad(error_matrix: np.ndarray):
        return np.apply_along_axis(median_absolute_deviation, 0, error_matrix)

    def _select_with_stream(self, population: Population, cases: CaseStream) -> Individual:
        candidates = one_individual_per_error_vector(population)

        ep = self.epsilon
        if isinstance(ep, bool) and ep:
            ep = self._epsilon_from_mad(population.all_error_vectors())

        t = time.time()
        for case in cases:
            if len(candidates) <= 1:
                break

            errors_this_case = [i.error_vector[case] for i in candidates]
            best_val_for_case = min(errors_this_case)

            max_error = best_val_for_case
            if isinstance(ep, np.ndarray):
                max_error += ep[case]
            elif isinstance(ep, (float, int, np.int64, np.float64)):
                max_error += ep

            candidates = [
                i for i in candidates if i.error_vector[case] <= max_error]
        population.logs['lexicase_runtime'] += time.time() - t

        return choice(candidates)

    def select_one(self, population: Population) -> Individual:
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        cases = CaseStream(len(population[0].error_vector))
        return self._select_with_stream(population, cases)


class Elite(Selector):
    """Returns the best N individuals by total error."""

    def select_one(self, population: Population) -> Individual:
        """Return single individual from population.

        Parameters
        ----------
        population
            A Population of Individuals.

        Returns
        -------
        Individual
            The selected Individual.

        """
        return population.best()

    def select(self, population: Population, n: int = 1) -> Sequence[Individual]:
        """Return `n` individuals from the population.

        Parameters
        ----------
        population
            A Population of Individuals.
        n : int
            The number of parents to select from the population. Default is 1.

        Returns
        -------
        Sequence[Individual]
            The selected Individuals.

        """
        return population.best_n(n)
