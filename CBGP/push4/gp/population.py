from collections.abc import Sequence
from bisect import insort_left
from typing import Callable

import numpy as np
import pickle
from multiprocessing import Pool
from functools import partial

from push4.gp.individual import Individual
from push4.lang.dag import Dag
from copy import copy
import time


def _eval_indiv(indiv: Individual, error_fn: Callable[[Dag], np.array]):
    indiv.error_vector = error_fn(indiv.program)
    return indiv


def preselect(population):
    """Preselect one individual per distinct error vector.

    Crucial for avoiding the worst case runtime of lexicase selection but
    does not impact the behavior of which individual gets selected.
    """
    population_list = list(copy(population))

    error_matrix = []
    error_vector_hashes = []
    candidates = {}

    for individual in population_list:
        error_vector_hash = hash(individual.error_vector_bytes)
        if error_vector_hash not in error_vector_hashes:
            error_vector_hashes.append(error_vector_hash)
            candidates[error_vector_hash] = [individual]
            error_matrix.append(individual.error_vector)
        else:
            candidates[error_vector_hash] += [individual]

    return np.array(error_matrix), error_vector_hashes, candidates


class Population(Sequence):
    """A sequence of Individuals kept in sorted order, with respect to their total errors."""

    __slots__ = ["unevaluated", "evaluated", "nondom_set", "logs"]

    def __init__(self, individuals: list = None):
        self.unevaluated = []
        self.evaluated = []
        self.nondom_set = []
        self.logs = {
            'lexicase_runtime': 0
        }

        if individuals is not None:
            for el in individuals:
                self.add(el)

    def __len__(self):
        return len(self.evaluated) + len(self.unevaluated)

    def __getitem__(self, key: int) -> Individual:
        if key < len(self.evaluated):
            return self.evaluated[key]
        return self.unevaluated[key - len(self.evaluated)]

    def add(self, individual: Individual):
        """Add an Individual to the population."""
        if individual.total_error is None:
            self.unevaluated.append(individual)
        else:
            insort_left(self.evaluated, individual)
        return self

    def best(self):
        """Return the best n individual in the population."""
        return self.evaluated[0]

    def best_n(self, n: int):
        """Return the best n individuals in the population."""
        return self.evaluated[:n]

    def p_evaluate(self, error_fn, pool: Pool):
        """Evaluate all unevaluated individuals in the population in parallel."""
        func = partial(_eval_indiv, error_fn=error_fn)
        for individual in pool.imap_unordered(func, self.unevaluated):
            insort_left(self.evaluated, individual)
        self.unevaluated = []

    def evaluate(self, error_fn: Callable[[Dag], np.array], downsample_rate=1., alpha=1.):
        """Evaluate all unevaluated individuals in the population."""
        downsample_size = int(downsample_rate * len(self.unevaluated))
        downsample_idx = np.random.choice(
            np.arange(len(self.unevaluated)), size=downsample_size, replace=False)

        for i, individual in enumerate(self.unevaluated):
            if i in downsample_idx:
                individual = _eval_indiv(individual, error_fn)
                insort_left(self.evaluated, individual)
        self.unevaluated = []

        error_matrix, error_vector_hashes, candidates = preselect(self)

        t = time.time()
        # generate dominate set
        best_err = np.min(error_matrix, axis=0)
        n_cand, n_cases = error_matrix.shape

        err_is_best = (error_matrix <= best_err).astype(float)
        n_best = np.sum(err_is_best, axis=1)

        unchecked = np.argsort(n_best)
        dom_set = []
        total_comparisons = 0

        while len(unchecked) > 0:
            idx = unchecked[0]

            cur_n_best = n_best[idx]
            to_compare = np.arange(n_cand)[n_best <= cur_n_best]
            # remove self
            to_compare = to_compare[to_compare != idx]
            # remove already dominated cand
            to_compare = to_compare[np.logical_not(
                np.isin(to_compare, dom_set))]

            if len(to_compare) > 0:
                total_comparisons += len(to_compare)

                cur_err = error_matrix[idx]
                to_compare_err = error_matrix[to_compare]

                cur_err_is_best = err_is_best[idx]
                to_compare_err_is_best = err_is_best[to_compare]

                # A - current; B - to compare
                # tie on some best and B is better on something
                cond1 = (np.sum(cur_err_is_best * to_compare_err_is_best,
                                axis=1) *
                         np.sum((to_compare_err < cur_err), axis=1)
                         ) > 0

                # B is better on something with best error
                cond2 = np.sum(to_compare_err_is_best *
                               (1-cur_err_is_best), axis=1) > 0

                dom_set += list(to_compare[np.logical_not(np.logical_or(cond1, cond2))])

            unchecked = unchecked[unchecked != idx]  # remove self
            unchecked = unchecked[np.logical_not(np.isin(unchecked, dom_set))]

        nondom_set = np.arange(n_cand)
        nondom_set = nondom_set[np.logical_not(np.isin(nondom_set, dom_set))]
        n_best_nondom = n_best[nondom_set]

        best_each_case = np.array(
            [error_matrix[nondom_set, i] == best_err[i] for i in range(n_cases)]).astype(float)  # (n_cases, nondom set)
        p_each_case = best_each_case * n_best_nondom
        p_each_case = p_each_case / np.sum(p_each_case, axis=1, keepdims=True)
        p = np.sum(p_each_case, axis=0)

        # normalize
        p = p/np.sum(p)

        # manipulate
        p = np.power(p, alpha)
        p = p/np.sum(p)

        self.nondom_set = ([candidates[error_vector_hashes[i]]
                           for i in nondom_set], p)

        self.logs['lexiprob_runtime'] = time.time() - t
        self.logs['lexiprob_comparisons'] = total_comparisons

    def all_error_vectors(self):
        """2D array containing all Individuals' error vectors."""
        return np.vstack([i.error_vector for i in self.evaluated])

    def all_total_errors(self):
        """1D array containing all Individuals' total errors."""
        return np.array([i.total_error for i in self.evaluated])

    def median_error(self):
        """Median total error in the population."""
        return np.median(self.all_total_errors())

    def error_diversity(self):
        """Proportion of unique error vectors."""
        return len(np.unique(self.all_error_vectors(), axis=0)) / float(len(self))

    def genome_diversity(self):
        """Proportion of unique genomes."""
        unq = set([pickle.dumps(i.genome) for i in self])
        return len(unq) / float(len(self))

    def program_diversity(self):
        """Proportion of unique programs."""
        unq = set([pickle.dumps(i.get_program().code) for i in self])
        return len(unq) / float(len(self))
