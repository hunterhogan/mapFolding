"""Count with `mapFolding.algorithms.pNxN` and Numba compilation of selected functions.

(AI generated docstring)

`mapFolding.kitAST.numba.kitNumba.makeNumbaModule` generates this module.
Regenerate the module instead of editing it.

Contents
--------
Functions
    count
        Count the ways to insert the leaves from `leafStart` to the last leaf.
    countBits
        Count the set bits of `gaps`.
    deduplicateFolds
        Keep one partial folding from each orbit of the core, weighted by the size of the orbit.
    doTheNeedful
        Count the foldings of an n × n map.
    generateFolds
        Make each partial folding that inserts the leaves from `leafStart` to `leafStop`.
    getGaps
        Make the set of leaves below which `leaf` can be inserted.
    makeEdges
        Make the table of the leaf that each crease connects to each leaf.
    makeLeafByCoordinate
        Number the leaves of an n × n map in a spiral that ends at the center.
    makeOrbit
        Make the set of partial foldings equivalent to `leafBelow` under the symmetries of the core.
    makePermutations
        Make the permutation of the leaves for each of the eight symmetries of the square.
"""
from __future__ import annotations

from itertools import repeat
from mapFolding.beDRY import dotProduct, getTotalLeaves
from mapFolding.dataStructures import make_zeros
from multiprocessing import Pool
from numba import jit
from operator import itemgetter
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
    from collections.abc import Iterable
    from mapFolding.theTypes import 形Array1Dint64, 形Array2Dint64
    type 形ArrayGaps1D = numpy.ndarray[tuple[int], numpy.dtype[numpy.uint64]]

@jit(boundscheck=False, cache=True, error_model='numpy', forceinline=True, inline='always', nopython=True, parallel=False)
def count(leafBelow: 形Array1Dint64, leafStart: int, edges: 形Array2Dint64) -> int:
    """Count the ways to insert the leaves from `leafStart` to the last leaf.

    (AI generated docstring)

    Input Mutation
    --------------
    This function uses `leafBelow` as workspace. When the count is complete, the entries of the leaves
    before `leafStart` again describe the supplied partial folding.

    Parameters
    ----------
    leafBelow : 形Array1Dint64
        Partial folding as a circular stack: `leafBelow[leaf]` is the leaf directly below `leaf`. The
        leaves before `leafStart` are in the stack.
    leafStart : int
        First leaf to insert.
    edges : 形Array2Dint64
        Table from `makeEdges`.

    Returns
    -------
    total : int
        Number of foldings that extend the partial folding.
    """
    totalLeaves: int = leafBelow.size
    leafAbove: 形Array1Dint64 = make_zeros(totalLeaves, numpy.dtype(numpy.int64))
    boxOfGaps: 形ArrayGaps1D = make_zeros(totalLeaves, numpy.dtype(numpy.uint64))
    leaf: int = leafStart
    boxOfGaps[leaf] = getGaps(leafBelow, leaf, edges)
    total: int = 0
    while leafStart <= leaf:
        if leaf == totalLeaves - 1:
            total += countBits(boxOfGaps[leaf])
            boxOfGaps[leaf] = numpy.uint64(0)
        if boxOfGaps[leaf]:
            while not boxOfGaps[leaf] & numpy.uint64(1) << numpy.uint64(leafAbove[leaf]):
                leafAbove[leaf] += 1
            boxOfGaps[leaf] ^= numpy.uint64(1) << numpy.uint64(leafAbove[leaf])
            leafBelow[leaf] = leafBelow[leafAbove[leaf]]
            leafBelow[leafAbove[leaf]] = leaf
            leaf += 1
            leafAbove[leaf] = 0
            boxOfGaps[leaf] = getGaps(leafBelow, leaf, edges)
        else:
            leaf -= 1
            if leafStart <= leaf:
                leafBelow[leafAbove[leaf]] = leafBelow[leaf]
    return total

@jit(boundscheck=False, cache=True, error_model='numpy', forceinline=True, inline='always', nopython=True, parallel=False)
def countBits(gaps: numpy.uint64) -> int:
    """Count the set bits of `gaps`.

    (AI generated docstring)

    Parameters
    ----------
    gaps : numpy.uint64
        Set of leaves from `getGaps`.

    Returns
    -------
    total : int
        Number of leaves in `gaps`.
    """
    total: int = 0
    while gaps:
        gaps &= gaps - numpy.uint64(1)
        total += 1
    return total

def deduplicateFolds(boxOfFolds: list[形Array1Dint64], n: int) -> list[tuple[形Array1Dint64, int]]:
    """Keep one partial folding from each orbit of the core, weighted by the size of the orbit.

    (AI generated docstring)

    Parameters
    ----------
    boxOfFolds : list[形Array1Dint64]
        Partial foldings of the core from `generateFolds`.
    n : int
        Side length of the core.

    Returns
    -------
    boxOfFoldsWeighted : list[tuple[形Array1Dint64, int]]
        Each representative partial folding with the number of partial foldings in its orbit.
    """
    boxOfPermutations: 形Array2Dint64 = makePermutations(n)
    boxOfFoldsSeen: set[tuple[int, ...]] = set()
    boxOfFoldsWeighted: list[tuple[形Array1Dint64, int]] = []
    次Folding: int = 0
    while 次Folding < len(boxOfFolds):
        leafBelow: 形Array1Dint64 = boxOfFolds[次Folding]
        if tuple(map(int, leafBelow[:boxOfPermutations.shape[1]])) not in boxOfFoldsSeen:
            orbit: set[tuple[int, ...]] = makeOrbit(leafBelow, boxOfPermutations)
            boxOfFoldsWeighted.append((leafBelow, len(orbit)))
            boxOfFoldsSeen.update(orbit)
        次Folding += 1
    return boxOfFoldsWeighted

@jit(boundscheck=False, cache=True, error_model='numpy', forceinline=True, inline='always', nopython=True, parallel=False)
def generateFolds(leafBelow: 形Array1Dint64, leafStart: int, edges: 形Array2Dint64, leafStop: int) -> list[形Array1Dint64]:
    """Make each partial folding that inserts the leaves from `leafStart` to `leafStop`.

    (AI generated docstring)

    Input Mutation
    --------------
    This function uses `leafBelow` as workspace, as `count` does.

    Parameters
    ----------
    leafBelow : 形Array1Dint64
        Partial folding as a circular stack. The leaves before `leafStart` are in the stack.
    leafStart : int
        First leaf to insert.
    edges : 形Array2Dint64
        Table from `makeEdges`.
    leafStop : int
        Leaf after the last leaf to insert.

    Returns
    -------
    boxOfFolds : list[形Array1Dint64]
        Copy of `leafBelow` for each partial folding of the leaves before `leafStop`.
    """
    leafAbove: 形Array1Dint64 = make_zeros(leafBelow.size, numpy.dtype(numpy.int64))
    boxOfGaps: 形ArrayGaps1D = make_zeros(leafBelow.size, numpy.dtype(numpy.uint64))
    boxOfFolds: list[形Array1Dint64] = []
    leaf: int = leafStart
    boxOfGaps[leaf] = getGaps(leafBelow, leaf, edges)
    while leafStart <= leaf:
        if boxOfGaps[leaf]:
            while not boxOfGaps[leaf] & numpy.uint64(1) << numpy.uint64(leafAbove[leaf]):
                leafAbove[leaf] += 1
            boxOfGaps[leaf] ^= numpy.uint64(1) << numpy.uint64(leafAbove[leaf])
            leafBelow[leaf] = leafBelow[leafAbove[leaf]]
            leafBelow[leafAbove[leaf]] = leaf
            leaf += 1
            if leaf == leafStop:
                boxOfFolds.append(leafBelow.copy())
                leaf -= 1
                leafBelow[leafAbove[leaf]] = leafBelow[leaf]
            else:
                leafAbove[leaf] = 0
                boxOfGaps[leaf] = getGaps(leafBelow, leaf, edges)
        else:
            leaf -= 1
            if leafStart <= leaf:
                leafBelow[leafAbove[leaf]] = leafBelow[leaf]
    return boxOfFolds

@jit(boundscheck=False, cache=True, error_model='numpy', forceinline=True, inline='always', nopython=True, parallel=False)
def getGaps(leafBelow: 形Array1Dint64, leaf: int, edges: 形Array2Dint64) -> numpy.uint64:
    """Make the set of leaves below which `leaf` can be inserted.

    (AI generated docstring)

    `leaf` can go directly below a leaf of the stack when the insertion keeps every crease of `leaf`
    to an earlier leaf from crossing another crease of the same kind.

    Parameters
    ----------
    leafBelow : 形Array1Dint64
        Partial folding as a circular stack of the leaves before `leaf`.
    leaf : int
        Leaf to insert.
    edges : 形Array2Dint64
        Table from `makeEdges`.

    Returns
    -------
    gaps : numpy.uint64
        Set of leaves: bit k is set when `leaf` can be inserted directly below leaf k.
    """
    gaps: numpy.uint64 = (numpy.uint64(1) << numpy.uint64(leaf)) - numpy.uint64(1)
    次Crease: int = 0
    while 次Crease < edges.shape[0] and gaps:
        leafConnectee: int = int(edges[次Crease, leaf])
        if leafConnectee < leaf:
            gapsCrease: numpy.uint64 = numpy.uint64(1) << numpy.uint64(leafConnectee)
            leafWalk: int = int(leafBelow[leafConnectee])
            while leafWalk != leafConnectee:
                leafAdjacent: int = int(edges[次Crease, leafWalk])
                if leafAdjacent < leaf:
                    leafWalk = leafAdjacent
                gapsCrease |= numpy.uint64(1) << numpy.uint64(leafWalk)
                leafWalk = int(leafBelow[leafWalk])
            gaps &= gapsCrease
        次Crease += 1
    return gaps

def makeEdges(leafByCoordinate: 形Array2Dint64) -> 形Array2Dint64:
    """Make the table of the leaf that each crease connects to each leaf.

    (AI generated docstring)

    Parameters
    ----------
    leafByCoordinate : 形Array2Dint64
        Leaf at each coordinate of the map, from `makeLeafByCoordinate`.

    Returns
    -------
    edges : 形Array2Dint64
        Array of shape `(4, totalLeaves)`. Row `2 * dimension + parity` holds, for each leaf, the leaf
        across the crease of that dimension and parity, or `totalLeaves` when the leaf has no such
        crease.
    """
    edges: 形Array2Dint64 = numpy.full((4, leafByCoordinate.size), leafByCoordinate.size, dtype=numpy.int64)
    次Dimension: int = 0
    while 次Dimension < 2:
        leafByAxis: 形Array2Dint64 = numpy.moveaxis(leafByCoordinate, 次Dimension, 0)
        次Parity: int = 0
        while 次Parity < 2:
            edges[2 * 次Dimension + 次Parity, leafByAxis[2 - 次Parity::2]] = leafByAxis[1 - 次Parity:-1:2]
            edges[2 * 次Dimension + 次Parity, leafByAxis[1 - 次Parity:-1:2]] = leafByAxis[2 - 次Parity::2]
            次Parity += 1
        次Dimension += 1
    return edges

def makeLeafByCoordinate(n: int) -> 形Array2Dint64:
    """Number the leaves of an n × n map in a spiral that ends at the center.

    (AI generated docstring)

    The last leaf is at a corner and leaf 0 is at the center, so for each core length k with the
    parity of n, the leaves before k² form the central k × k square.

    Parameters
    ----------
    n : int
        Side length of the map.

    Returns
    -------
    leafByCoordinate : 形Array2Dint64
        Array of shape `(n, n)` with the leaf at each coordinate.
    """
    totalLeaves: int = getTotalLeaves((n, n))
    leafByCoordinate: 形Array2Dint64 = numpy.full((n, n), totalLeaves, dtype=numpy.int64)
    x: int = 0
    y: int = 0
    stepX: int = 1
    stepY: int = 0
    leaf: int = totalLeaves - 1
    while 0 <= leaf:
        leafByCoordinate[x, y] = leaf
        xNext: int = x + stepX
        yNext: int = y + stepY
        if xNext < 0 or n <= xNext or yNext < 0 or (n <= yNext) or (leafByCoordinate[xNext, yNext] < totalLeaves):
            stepX, stepY = (-stepY, stepX)
            x += stepX
            y += stepY
        else:
            x, y = (xNext, yNext)
        leaf -= 1
    return leafByCoordinate

def makeOrbit(leafBelow: 形Array1Dint64, boxOfPermutations: 形Array2Dint64) -> set[tuple[int, ...]]:
    """Make the set of partial foldings equivalent to `leafBelow` under the symmetries of the core.

    (AI generated docstring)

    The symmetries are the eight symmetries of the square in `boxOfPermutations`, each with and
    without the reversal of the stack.

    Parameters
    ----------
    leafBelow : 形Array1Dint64
        Partial folding of the core as a circular stack.
    boxOfPermutations : 形Array2Dint64
        Permutations of the leaves of the core, from `makePermutations`.

    Returns
    -------
    orbit : set[tuple[int, ...]]
        Each equivalent partial folding, as the entries of `leafBelow` for the leaves of the core.
    """
    orbit: set[tuple[int, ...]] = set()
    次Permutation: int = 0
    while 次Permutation < boxOfPermutations.shape[0]:
        permutation: 形Array1Dint64 = boxOfPermutations[次Permutation]
        leafBelowPermuted: 形Array1Dint64 = numpy.empty(permutation.size, dtype=numpy.int64)
        leafBelowPermuted[permutation] = permutation[leafBelow[:permutation.size]]
        leafAbovePermuted: 形Array1Dint64 = numpy.empty(permutation.size, dtype=numpy.int64)
        leafAbovePermuted[leafBelowPermuted] = numpy.arange(permutation.size)
        orbit.add(tuple(map(int, leafBelowPermuted)))
        orbit.add(tuple(map(int, leafAbovePermuted)))
        次Permutation += 1
    return orbit

def makePermutations(n: int) -> 形Array2Dint64:
    """Make the permutation of the leaves for each of the eight symmetries of the square.

    (AI generated docstring)

    Parameters
    ----------
    n : int
        Side length of the square.

    Returns
    -------
    boxOfPermutations : 形Array2Dint64
        Array of shape `(8, n * n)`. Each row maps each leaf to its image under one rotation, with or
        without a reflection.
    """
    leafByCoordinate: 形Array2Dint64 = makeLeafByCoordinate(n)
    boxOfPermutations: 形Array2Dint64 = numpy.empty((8, getTotalLeaves((n, n))), dtype=numpy.int64)
    次Rotation: int = 0
    while 次Rotation < 4:
        boxOfPermutations[2 * 次Rotation, leafByCoordinate.ravel()] = numpy.rot90(leafByCoordinate, 次Rotation).ravel()
        boxOfPermutations[2 * 次Rotation + 1, leafByCoordinate.ravel()] = numpy.flip(numpy.rot90(leafByCoordinate, 次Rotation), axis=0).ravel()
        次Rotation += 1
    return boxOfPermutations

def doTheNeedful(n: int, *, workersMaximum: int = 1) -> int:
    """Count the foldings of an n × n map.

    (AI generated docstring)

    You can use this function to compute A001418(n). Leaves 0, 1, and 2 start the stack in a fixed
    order, and the factor `totalLeaves` restores the other positions of the stack. For n ≤ 5, the
    factor 2 restores the other order of leaves 1 and 2. For n ≥ 6, each folding of the central core
    represents its orbit, and for n ≥ 7 each representative is extended to 24 leaves before the count.

    Parameters
    ----------
    n : int
        Side length of the map.
    workersMaximum : int = 1
        Number of processes that count the representatives. A value of 1 or less counts in the calling
        process.

    Returns
    -------
    totalFolds : int
        Number of foldings of the n × n map.
    """
    if n == 1:
        totalFolds: int = 1
    elif n == 2:
        totalFolds = 8
    else:
        totalLeaves: int = getTotalLeaves((n, n))
        edges: 形Array2Dint64 = makeEdges(makeLeafByCoordinate(n))
        leafBelow: 形Array1Dint64 = make_zeros(totalLeaves, numpy.dtype(numpy.int64))
        leafBelow[:3] = (1, 2, 0)
        if n <= 5:
            totalFolds = 2 * totalLeaves * count(leafBelow, 3, edges)
        else:
            if n % 2 == 0:
                coreLength: int = 4
            else:
                coreLength = 3
            leafStart: int = getTotalLeaves((coreLength, coreLength))
            boxOfFolds: list[tuple[形Array1Dint64, int]] = deduplicateFolds(generateFolds(leafBelow, 3, edges, leafStart), coreLength)
            if 7 <= n:
                boxOfFoldsExpanded: list[tuple[形Array1Dint64, int]] = []
                次Folding: int = 0
                while 次Folding < len(boxOfFolds):
                    leafBelow, multiplicity = boxOfFolds[次Folding]
                    boxOfFoldsExpanded.extend(zip(generateFolds(leafBelow, leafStart, edges, 24), repeat(multiplicity)))
                    次Folding += 1
                boxOfFolds = boxOfFoldsExpanded
                leafStart = 24
            if 1 < workersMaximum:
                with Pool(processes=workersMaximum) as pool:
                    boxOfCounts: Iterable[int] = pool.starmap(count, zip(map(itemgetter(0), boxOfFolds), repeat(leafStart), repeat(edges)))
            else:
                boxOfCounts = map(count, map(itemgetter(0), boxOfFolds), repeat(leafStart), repeat(edges))
            totalFolds = totalLeaves * dotProduct(map(itemgetter(1), boxOfFolds), boxOfCounts, strict=True)
    return totalFolds
