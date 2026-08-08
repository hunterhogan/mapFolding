# DEVELOPMENT module.
# pyright: reportUnusedVariable=false
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""
from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import pairwise
from mapFolding._e import getLeafDomain
from mapFolding._e._2上nDimensional.reduceIt import listFunctionsReduction2上nDimensional, listFunctionsReductionQuick2上nDimensional
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

def makeDescendants(folding: Folding, n: int, position: int) -> EliminationState:
	"""Process a single folding and return resulting state."""
	state: EliminationState = EliminationState(makeMapShape('A001417', n)
		, listFunctionsReduction=listFunctionsReduction2上nDimensional
		, listFunctionsReductionQuick=listFunctionsReductionQuick2上nDimensional)
	state.listPermutationSpace.append(PermutationSpace(getDictionaryLeafOptions(state)))
	state.reduceAllPermutationSpace(quick=True)

	for r, k in tqdm(pairwise(reversed(folding[2:None])), total=len(folding) - 3, disable=False, position=position, leave=False):
		state = excludeLeaf_rBeforeLeaf_k(state, k, r, getLeafDomain(state, k), getLeafDomain(state, r))

	state.removeCreaseViolations().reduceAllPermutationSpace()

	return state

def makeAlbum2上nDimensional吗(n: int, workersMaximum: int) -> EliminationState:
	"""Construct album `n`."""
	album: Iterable[Folding] = readAlbum(makePathFilenameFolds(makeMapShape('A001417', n - 1), pathAlbum, suffix='.album'))

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		listClaimTickets: list[Future[EliminationState]] = [
			concurrencyManager.submit(makeDescendants, folding, n, position % workersMaximum + 1)
			for position, folding in enumerate(album)
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
