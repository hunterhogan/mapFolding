# https://codegolf.stackexchange.com/questions/276709/count-the-possible-folds-of-an-n%c2%b2-grid
from __future__ import annotations

from itertools import repeat
from mapFolding.beDRY import dotProduct, getTotalLeaves
from mapFolding.dataStructures import make_zeros
from multiprocessing import Pool
from operator import itemgetter
from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Iterable
	from mapFolding.theTypes import 形Array1Dint64, 形Array2Dint64

	type 形ArrayGaps1D = numpy.ndarray[tuple[int], numpy.dtype[numpy.uint64]]

def count(leafBelow: 形Array1Dint64, leafStart: int, edges: 形Array2Dint64) -> int:
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
			while not boxOfGaps[leaf] & (numpy.uint64(1) << numpy.uint64(leafAbove[leaf])):
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

def countBits(gaps: numpy.uint64) -> int:
	total: int = 0
	while gaps:
		gaps &= gaps - numpy.uint64(1)
		total += 1
	return total

def deduplicateFolds(boxOfFolds: list[形Array1Dint64], n: int) -> list[tuple[形Array1Dint64, int]]:
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

def generateFolds(leafBelow: 形Array1Dint64, leafStart: int, edges: 形Array2Dint64, leafStop: int) -> list[形Array1Dint64]:
	leafAbove: 形Array1Dint64 = make_zeros(leafBelow.size, numpy.dtype(numpy.int64))
	boxOfGaps: 形ArrayGaps1D = make_zeros(leafBelow.size, numpy.dtype(numpy.uint64))
	boxOfFolds: list[形Array1Dint64] = []
	leaf: int = leafStart
	boxOfGaps[leaf] = getGaps(leafBelow, leaf, edges)
	while leafStart <= leaf:
		if boxOfGaps[leaf]:
			while not boxOfGaps[leaf] & (numpy.uint64(1) << numpy.uint64(leafAbove[leaf])):
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

def getGaps(leafBelow: 形Array1Dint64, leaf: int, edges: 形Array2Dint64) -> numpy.uint64:
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
		if (xNext < 0 or n <= xNext or yNext < 0 or n <= yNext
			or leafByCoordinate[xNext, yNext] < totalLeaves):
			stepX, stepY = -stepY, stepX
			x += stepX
			y += stepY
		else:
			x, y = xNext, yNext
		leaf -= 1
	return leafByCoordinate

def makeOrbit(leafBelow: 形Array1Dint64, boxOfPermutations: 形Array2Dint64) -> set[tuple[int, ...]]:
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
	leafByCoordinate: 形Array2Dint64 = makeLeafByCoordinate(n)
	boxOfPermutations: 形Array2Dint64 = numpy.empty((8, getTotalLeaves((n, n))), dtype=numpy.int64)
	次Rotation: int = 0
	while 次Rotation < 4:
		boxOfPermutations[2 * 次Rotation, leafByCoordinate.ravel()] = numpy.rot90(leafByCoordinate, 次Rotation).ravel()
		boxOfPermutations[2 * 次Rotation + 1, leafByCoordinate.ravel()] = numpy.flip(numpy.rot90(leafByCoordinate, 次Rotation), axis=0).ravel()
		次Rotation += 1
	return boxOfPermutations

def doTheNeedful(n: int, *, workersMaximum: int = 1) -> int:
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
