"""The :mod:`variation` module defines classes for variation operators.

Variation operators (aka genetic operators) are used in evolutionary/genetic
algorithms to create "child" genomes from "parent" genomes.

"""
from abc import ABC, abstractmethod
from typing import Sequence, Union, Tuple
import math
from copy import copy

from numpy.random import random, choice
from pyrsistent import pvector, PVector

from push4.collections import DiscreteProbDistrib
from push4.gp.individual import Genome
from push4.gp.spawn import Spawner
from push4.lang.expr import Expression


class VariationOperator(ABC):
    """Base class of all VariationOperators.

    Parameters
    ----------
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.

    Attributes
    ----------
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.

    """

    def __init__(self, num_parents: int):
        self.num_parents = num_parents

    def checknum_parents(self, parents: Sequence[Genome]):
        """Raise error if given too few parents.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.

        """
        if not len(parents) >= self.num_parents:
            raise ValueError("Variation operator given {a} parents. Expected {e}.".format(
                a=len(parents),
                e=self.num_parents)
            )

    @abstractmethod
    def produce(self, parents: Sequence[Genome], spawner: Spawner) -> Genome:
        """Produce a child Genome from parent Genomes and optional GenomeSpawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A GeneSpawner that can be used to produce new genes (aka Atoms).

        """
        pass


class VariationSet(VariationOperator):
    """A collection of VariationOperator and how frequently to use them."""

    def __init__(self, operators: Sequence[Tuple[VariationOperator, float]]):
        # @TODO: Figure out how to avoid selecting extra parents.
        num_parents_needed = max([op.num_parents for op, _ in operators])
        super().__init__(num_parents_needed)
        self.ops = DiscreteProbDistrib()
        for op, prob in operators:
            self.ops.add(op, prob)

    def produce(self, parents: Sequence[Genome], spawner: Spawner) -> Genome:
        """Produce a child Genome from parent Genomes and optional GenomeSpawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A GeneSpawner that can be used to produce new genes (aka Atoms).

        """
        op = self.ops.sample()
        return op.produce(parents, spawner)


class VariationPipeline(VariationOperator):
    """Variation operator that sequentially applies multiple others variation operators.

    Parameters
    ----------
    operators : list of VariationOperators
        A list of operators to apply in order to produce the child Genome.

    Attributes
    ----------
    operators : list of VariationOperators
        A list of operators to apply in order to produce the child Genome.

    """

    def __init__(self, operators: Sequence[VariationOperator]):
        num_parents_needed = max([op.num_parents for op in operators])
        super().__init__(num_parents_needed)
        self.operators = operators

    def produce(self, parents: Sequence[Genome], spawner: Spawner) -> Genome:
        """Produce a child Genome from parent Genomes and optional GenomeSpawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A GeneSpawner that can be used to produce new genes (aka Atoms).

        """
        self.checknum_parents(parents)
        child = copy(parents[0])
        for op in self.operators:
            child = op.produce([child] + parents[1:], spawner)
        return child


# Utilities

def _gaussian_noise_factor():
    """Return Gaussian noise of mean 0, std dev 1.

    Returns
    --------
    Float samples from Gaussian distribution.

    Examples
    --------
    >>> _gaussian_noise_factor()
    1.43412557975
    >>> _gaussian_noise_factor()
    -0.0410900866765

    """
    return math.sqrt(-2.0 * math.log(random())) * math.cos(2.0 * math.pi * random())


# Mutations

class DeletionMutation(VariationOperator):
    """Uniformly randomly removes some Atoms from parent.

    Parameters
    ----------
    deletion_rate : float
        The probablility of removing any given Atom in the parent Genome.
        Default is 0.01.

    Attributes
    ----------
    rate : float
        The probablility of removing any given Atom in the parent Genome.
        Default is 0.01.
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.

    """

    def __init__(self, deletion_rate: float = 0.01):
        super().__init__(1)
        self.rate = deletion_rate

    def produce(self, parents: Sequence[Genome], spawner: Spawner) -> Genome:
        """Produce a child Genome from parent Genomes and optional Spawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A GeneSpawner that can be used to produce new genes (aka Atoms).

        """
        self.checknum_parents(parents)
        new_genome = pvector()
        for gene in parents[0]:
            if random() < self.rate:
                continue
            new_genome = new_genome.append(gene)
        return pvector(new_genome)


class AdditionMutation(VariationOperator):
    """Uniformly randomly adds some Atoms to parent.

    Parameters
    ----------
    addition_rate : float
        The probability of adding a new Atom at any given point in the parent
        Genome. Default is 0.01.

    Attributes
    ----------
    rate : float
        The probability of adding a new Atom at any given point in the parent
        Genome. Default is 0.01.
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.

    """

    def __init__(self, addition_rate: float = 0.01):
        super().__init__(1)
        self.rate = addition_rate

    def produce(self, parents: Sequence[Genome], spawner: Spawner) -> Genome:
        """Produce a child Genome from parent Genomes and optional Spawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A Spawner that can be used to produce new genes (aka Atoms).

        """
        self.checknum_parents(parents)
        new_genome = pvector()
        for gene in parents[0]:
            if random() < self.rate:
                new_genome = new_genome.append(spawner.spawn_gene())
            new_genome = new_genome.append(gene)
        if random() < self.rate:
            new_genome = new_genome.append(spawner.spawn_gene())
        return pvector(new_genome)


def umad(addition_rate: float, deletion_rate: float):
    return VariationPipeline([
        AdditionMutation(addition_rate),
        DeletionMutation(deletion_rate)
    ])


size_neutral_umad = umad(0.09, 0.0826)
shrinking_umad = umad(0.09, 0.1)
growing_umad = umad(0.09, 0.0652)


# Recombination

class Alternation(VariationOperator):
    """Uniformly alternates between the two parent genomes.

    Parameters
    ----------
    alternation_rate : float, optional (default=0.01)
        The probability of switching which parent program elements are being
        copied from. Must be 0 <= rate <= 1. Defaults to 0.1.
    alignment_deviation : int, optional (default=10)
        The standard deviation of how far alternation may jump between indices
        when switching between parents.

    Attributes
    ----------
    alternation_rate : float, optional (default=0.01)
        The probability of switching which parent program elements are being
        copied from. Must be 0 <= rate <= 1. Defaults to 0.1.
    alignment_deviation : int, optional (default=10)
        The standard deviation of how far alternation may jump between indices
        when switching between parents.
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.

    """

    def __init__(self, alternation_rate=0.01, alignment_deviation=10):
        super().__init__(2)
        self.alternation_rate = alternation_rate
        self.alignment_deviation = alignment_deviation

    def produce(self, parents: Sequence[Genome], spawner: Spawner = None) -> Genome:
        """Produce a child Genome from parent Genomes and optional Spawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A Spawner that can be used to produce new genes (aka Atoms).

        """
        self.checknum_parents(parents)
        gn1 = copy(parents[0])
        gn2 = copy(parents[1])
        new_genome = pvector()
        # Random pick which parent to start from
        use_parent_1 = choice([True, False])
        loop_times = len(gn1)
        if not use_parent_1:
            loop_times = len(gn2)
        i = 0
        while i < loop_times:
            if random() < self.alternation_rate:
                # Switch which parent we are pulling genes from
                i += round(self.alignment_deviation * _gaussian_noise_factor())
                i = int(max(0, i))
                use_parent_1 = not use_parent_1
            else:
                # Pull gene from parent
                if use_parent_1:
                    new_genome = new_genome.append(gn1[i])
                else:
                    new_genome = new_genome.append(gn2[i])
                i = int(i + 1)
            # Change loop stop condition
            loop_times = len(gn1)
            if not use_parent_1:
                loop_times = len(gn2)
        return new_genome


# Other

class Genesis(VariationOperator):
    """Creates an entirely new (and random) genome.

    Parameters
    ----------
    size
        The child genome will contain this many Atoms if size is an integer.
        If size is a pair of integers, the genome will be of a random
        size in the range of the two integers.

    Attributes
    ----------
    size
        The child genome will contain this many Atoms if size is an integer.
        If size is a pair of integers, the genome will be of a random
        size in the range of the two integers.
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.

    """

    def __init__(self, *, size: Tuple[int, int]):
        super().__init__(0)
        self.size = size

    def produce(self, parents: Sequence[Genome], spawner: Spawner) -> Genome:
        """Produce a child Genome from parent Genomes and optional GenomeSpawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A Spawner that can be used to produce new genes (aka Atoms).

        """
        return spawner.spawn_genome(*self.size)


class Cloning(VariationOperator):
    """Clones the parent genome.

    Attributes
    ----------
    num_parents : int
        Number of parent Genomes the operator needs to produce a child
        Individual.

    """

    def __init__(self):
        super().__init__(1)

    def produce(self, parents: Sequence[Genome], spawner: Spawner = None) -> Genome:
        """Produce a child Genome from parent Genomes and optional Spawner.

        Parameters
        ----------
        parents
            A list of parent Genomes given to the operator.
        spawner
            A Spawner that can be used to produce new genes (aka Atoms).

        """
        return parents[0]
