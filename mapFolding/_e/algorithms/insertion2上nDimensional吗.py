# DEVELOPMENT module.
# pyright: reportUnusedVariable=false
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""
from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import pairwise
from mapFolding._e import getLeafDomain
from mapFolding._e._2上nDimensional import 首一
from mapFolding._e._2上nDimensional.pinIt import pin首beans
from mapFolding._e._2上nDimensional.reduceIt import boxOfFunctionsReduction2上nDimensional
from mapFolding._e.dataBaskets import EliminationState
from mapFolding._e.pinIt import atPileExcludeLeaf_inboxOfPermutationSpace, excludeLeaf_rBeforeLeaf_k
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

def makeDescendants(folding: Folding, state: EliminationState, position: int) -> EliminationState:
	"""Process a single folding and return resulting state."""
	for pile, (r, k) in tqdm(enumerate(pairwise(reversed(folding[2:None])), start=1), total=len(folding) - 3, disable=False, position=position, leave=False):
		if r == 2:
			continue
		domain_r: Iterable[int] = getLeafDomain(state, r)
		pilesForOthers: set[int] = set(range(len(folding) - pile)).intersection(domain_r)
		for pile in pilesForOthers:
			state.boxOfPermutationSpace = atPileExcludeLeaf_inboxOfPermutationSpace(state.boxOfPermutationSpace, pile, r)

		state = excludeLeaf_rBeforeLeaf_k(state, k, r, getLeafDomain(state, k), domain_r)

		if r == 首一(state.dimensionsTotal):
			state = pin首beans(state, CPUlimit=True)

	state.removeCreaseViolations().reduceAllPermutationSpace()

	return state

def makeAlbum2上nDimensional吗(state: EliminationState, workersMaximum: int) -> EliminationState:
	"""Construct album `n`."""
	album: Iterable[Folding] = readAlbum(makePathFilenameFolds(makeMapShape('A001417', state.dimensionsTotal - 1), pathAlbum, suffix='.album'))

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		boxOfClaimTickets: list[Future[EliminationState]] = []
		for position, folding, in enumerate(album):
			stateCopy: EliminationState = EliminationState(state.mapShape, state.boxOfPermutationSpace.copy()
						, boxOfFunctionsReduction=boxOfFunctionsReduction2上nDimensional
					)
			boxOfClaimTickets.append(concurrencyManager.submit(makeDescendants, folding, stateCopy, position % workersMaximum + 1))

		state.boxOfPermutationSpace = []

		for claimTicket in tqdm(as_completed(boxOfClaimTickets), total=len(boxOfClaimTickets), disable=False, desc='for folding in album'):
			sherpa: EliminationState = claimTicket.result()
			state.boxOfPermutationSpace.extend(sherpa.boxOfPermutationSpace)
			state.boxOfFolding.extend(sherpa.boxOfFolding)

	return state

def recordAlbum2上nDimensional吗(state: EliminationState) -> Path:
	pathFilenameAlbum: Path = makePathFilenameFolds(state.mapShape, pathAlbum, suffix='.album')
	writeAlbum(sorted(state.boxOfFolding), pathFilenameAlbum)
	return pathFilenameAlbum
