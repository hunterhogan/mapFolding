# DEVELOPMENT module.
# pyright: reportUnusedVariable=false, reportUnusedImport=false
# ruff: file-ignore[commented-out-code, print]
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""

from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import partial
from humpy_cytoolz import valmap as mapLeaves
from hunterMakesPy import decreasing, inclusive, zeroIndexed
from itertools import chain, combinations, repeat
from mapFolding._e import leafOptionsAND, leafOrigin, makeLeafAntiOptions, pileOrigin
from mapFolding._e._2上nDimensional import 零, 首零
from mapFolding._e._2上nDimensional.pinIt import listFunctionsReduction2上nDimensional, pinPilesAtEnds
from mapFolding._e.algorithms.constraintPropagation import doTheNeedful
from mapFolding._e.algorithms.iff import creaseViolation吗
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.pileOptions import getDictionaryLeafOptions
from mapFolding.beDRY import defineProcessorLimit, getLeavesTotal
from mapFolding.kitFilesystem import makePathFilenameFolds, readAlbum, streamAlbum, writeAlbum
from mapFolding.oeis import getValuesKnown, makeMapShape
from mapFolding.theSSOT import settingsPackage
from multiprocessing.pool import Pool
from operator import add, methodcaller
from pprint import pprint
from time import perf_counter
from tqdm.auto import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Collection, Iterable, Sequence
	from concurrent.futures import Future
	from hunterMakesPy.theTypes import Limitation
	from mapFolding._e.theTypes import Folding, Leaf, LeafOptions, LeafSpace, Pile, PinnedLeaves, UndeterminedPiles
	from pathlib import Path

pathAlbum: Path = settingsPackage.pathPackage / '_e' / '_development' / 'albums'

def processFolding(
	folding: Folding, pileLast: int, countPinnedLeaves: int, undeterminedPilesDescendants: UndeterminedPiles, mapShape: tuple[int, ...]
) -> EliminationState:
	"""Process a single folding and return resulting state."""
	state: EliminationState = EliminationState(mapShape)
	mergeUndeterminedPilesDescendants: Callable[[PermutationSpace], PermutationSpace] = methodcaller(
		'addMissingPileLeafSpace', undeterminedPilesDescendants
	)
	listIndicesPinned = getIndices(pileLast, countPinnedLeaves)

	listPermutationSpaceIncomplete: Iterable[PermutationSpace] = map(PermutationSpace, map(zip, listIndicesPinned, repeat(folding)))
	listPermutationSpace: Iterable[PermutationSpace] = map(mergeUndeterminedPilesDescendants, listPermutationSpaceIncomplete)  # ruff: ignore[unused-variable]
	state.listPermutationSpace.extend(map(mergeUndeterminedPilesDescendants, listPermutationSpaceIncomplete))

	return state.removeCreaseViolations().reduceAllPermutationSpace(listFunctionsReduction2上nDimensional).moveToListFolding()

def makeAlbum2上nDimensional吗(n: int, workersMaximum: int) -> EliminationState:
	"""Construct album `n`."""
	mapShape: tuple[int, ...] = makeMapShape('A001417', n - 1)
	listPinnedLeaves: tuple[int, ...] = tuple(range(getLeavesTotal(mapShape)))
	countPinnedLeaves = len(listPinnedLeaves) - 2
	pathFilenameAlbum: Path = makePathFilenameFolds(mapShape, pathAlbum, suffix='.album')
	album: Iterable[Folding] = readAlbum(pathFilenameAlbum)

	mapShape = makeMapShape('A001417', n)
	state: EliminationState = EliminationState(mapShape)
	undeterminedPilesDescendants: UndeterminedPiles = getDictionaryLeafOptions(state)
	leafAntiOptionsPinnedLeaves: LeafOptions = makeLeafAntiOptions(state.leavesTotal, listPinnedLeaves)
	undeterminedPilesDescendants = mapLeaves(partial(leafOptionsAND, leafAntiOptionsPinnedLeaves), undeterminedPilesDescendants)

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(
				processFolding, folding, state.pileLast, countPinnedLeaves, undeterminedPilesDescendants, state.mapShape
			)
			for folding in album
		]

		state.listPermutationSpace = []

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False, desc='for folding in album'):
			sherpa: EliminationState = claimTicket.result()
			state.listPermutationSpace.extend(sherpa.listPermutationSpace)
			state.listFolding.extend(sherpa.listFolding)

	return state

def getIndices(pileLast: int, countPinnedLeaves: int) -> Iterable[tuple[Pile, ...]]:
	pile0Pile1: tuple[Pile, Pile] = (pileOrigin, 零)
	combinationsIndices = combinations(range(2, pileLast), countPinnedLeaves)
	return map(add, repeat(pile0Pile1), combinationsIndices)

def recordAlbum2上nDimensional吗(state: EliminationState) -> Path:
	pathFilenameAlbum = makePathFilenameFolds(state.mapShape, pathAlbum, suffix='.album')
	writeAlbum(sorted(state.listFolding), pathFilenameAlbum)
	return pathFilenameAlbum

if __name__ == '__main__':
	CPUlimit: Limitation = -2
	workersMaximum: int = defineProcessorLimit(CPUlimit)
	state = EliminationState((2,) * 3)
	state = doTheNeedful(state, 1)
	print(state)
	start: float = perf_counter()
	# aa = makeAlbums2上nDimensional吗(2, nFinal, workersMaximum)
	print(f'{perf_counter() - start:.2f}')
	# cc = len(aa.read_text(encoding="utf-8").splitlines()) * 2
	# vv = getValuesKnown('A001517')
	# print(cc == vv[nFinal + inclusive], cc, vv[nFinal + inclusive])
