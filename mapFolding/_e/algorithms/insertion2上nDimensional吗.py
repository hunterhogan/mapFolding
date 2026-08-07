# DEVELOPMENT module.
# pyright: reportUnusedVariable=false
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""
from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import pairwise
from mapFolding._e._2上nDimensional.pinIt import listFunctionsReduction2上nDimensional, listFunctionsReductionQuick2上nDimensional
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.pileOptions import getDictionaryLeafOptions
from mapFolding._e.pinIt import excludeLeaf_rBeforeLeaf_k
from mapFolding.kitFilesystem import makePathFilenameFolds, readAlbum, writeAlbum
from mapFolding.oeis import makeMapShape
from mapFolding.theSSOT import settingsPackage
from tqdm.auto import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable
	from concurrent.futures import Future
	from mapFolding._e.theTypes import Folding
	from pathlib import Path

pathAlbum: Path = settingsPackage.pathPackage / '_e' / '_development' / 'albums'

# ruff: file-ignore[commented-out-code]
def makeDescendants(folding: Folding, n: int) -> EliminationState:
	"""Process a single folding and return resulting state."""
	state: EliminationState = EliminationState(makeMapShape('A001417', n)
		, listFunctionsReduction=listFunctionsReduction2上nDimensional
		, listFunctionsReductionQuick=listFunctionsReductionQuick2上nDimensional)
	state.listPermutationSpace.append(PermutationSpace(getDictionaryLeafOptions(state)))

	for r, k in pairwise(reversed(folding)):
		state = excludeLeaf_rBeforeLeaf_k(state, k, r)
		state.moveToListPinnedLeaves()

	# d = 5
	# folding[-1] == 首一(len(mapShape))
	# 首一(5), leaf8 domain = [15, 17, 19, 21, 23, 25, 27, 29]
	# pileLast = 31, which has leaf16 pinned, pile29 is the largest same-parity open pile for leaf8.
	# neg(零) + 首零(5) = 15

	return state

def makeAlbum2上nDimensional吗(n: int, workersMaximum: int) -> EliminationState:
	"""Construct album `n`."""
	album: Iterable[Folding] = readAlbum(makePathFilenameFolds(makeMapShape('A001417', n - 1), pathAlbum, suffix='.album'))

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(makeDescendants, folding, n)
			for folding in album
		]

		state: EliminationState = EliminationState(makeMapShape('A001417', n))

		for claimTicket in tqdm(as_completed(listClaimTickets), total=len(listClaimTickets), disable=False, desc='for folding in album'):
			sherpa: EliminationState = claimTicket.result()
			state.listPermutationSpace.extend(sherpa.listPermutationSpace)
			state.listPinnedLeaves.extend(sherpa.listPinnedLeaves)
			state.listFolding.extend(sherpa.listFolding)

	return state

def recordAlbum2上nDimensional吗(state: EliminationState) -> Path:
	pathFilenameAlbum: Path = makePathFilenameFolds(state.mapShape, pathAlbum, suffix='.album')
	writeAlbum(sorted(state.listFolding), pathFilenameAlbum)
	return pathFilenameAlbum
