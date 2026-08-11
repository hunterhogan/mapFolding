# DEVELOPMENT module.
# pyright: reportUnusedVariable=false
"""Find `groupsOfFolds` based on Sade's 1949 insertion algorithm."""
from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from humpy_cytoolz import keyfilter as filterPile
from itertools import pairwise
from mapFolding._e import getDomainLeaf, makeAntiChoicesLeaf
from mapFolding._e._2上nDimensional import 一, 二, 零
from mapFolding._e._2上nDimensional.pinIt import pin3beans2, pinLeavesDimension二, pinPilesAtEnds
from mapFolding._e._2上nDimensional.reduceIt import boxOfFunctionsReduction2上nDimensional
from mapFolding._e.dataBaskets import EliminationState, PermutationSpace
from mapFolding._e.filters import 是valid
from mapFolding._e.pinIt import atPileExcludeLeaf_inBulk, excludeLeaf_rBeforeLeaf_k
from mapFolding._e.reduceIt import reduceLeafSpace
from mapFolding.kitFilesystem import makePathFilenameFolds, readAlbum, writeAlbum
from mapFolding.oeis import makeMapShape
from mapFolding.theSSOT import settingsPackage
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems

if TYPE_CHECKING:
	from collections.abc import Iterable
	from concurrent.futures import Future
	from mapFolding._e.theTypes import ChoicesLeaf, Folding, Leaf, Pile
	from pathlib import Path

pathAlbum: Path = settingsPackage.pathPackage / '_e' / '_development' / 'albums'

def makeDescendants(folding: Folding, state: EliminationState, position: int) -> EliminationState:
	"""Process a single folding and return resulting state."""
	leaf一: Leaf = folding[-1]
	domain一: Iterable[Pile] = getDomainLeaf(state, leaf一)
	leaves: tuple[Leaf, ...] = tuple(reversed(folding[2:-1]))
	antiChoicesLeaf: ChoicesLeaf = makeAntiChoicesLeaf(state.leavesTotal, leaves)
	boxOfPermutationSpace: list[PermutationSpace] = []
	for pile in tqdm(domain一, total=len(domain一), disable=False, position=position, leave=False, desc=f"for pile in domain一 of folding {folding[2:6]}"):
		boxOfPermutationSpace.extend(state.boxOfPermutationSpace)
		state.boxOfPermutationSpace = []

		state = pinPilesAtEnds(state, 0, CPUlimit=True)
		state.pile = pile

		permutationSpace: PermutationSpace = state.boxOfPermutationSpace.pop()
		permutationSpace = permutationSpace.atPilePinLeaf(state.pile, leaf一)
		permutationSpace = reduceLeafSpace(permutationSpace, DOTitems(filterPile(state.pile.__lt__, permutationSpace.undeterminedPiles())), antiChoicesLeaf)
		state.boxOfPermutationSpace.append(permutationSpace)
		state.removeCreaseViolations().reduceAllPermutationSpace()

		leaves二: list[int] = [二 + 一, 二 + 一 + 零, 二 + 零, 二]  # [6, 7, 5, 4]
		for 次, (r, k) in enumerate(pairwise(leaves), start=1):
			if r == 2:
				continue
			if r == 3:
				state = pin3beans2(state, CPUlimit=True)
			if r in leaves二:
				leaves二.remove(r)
				if not leaves二:
					state = pinLeavesDimension二(state, CPUlimit=True)

			domain_r: Iterable[Pile] = tuple(filter(state.pile.__gt__, getDomainLeaf(state, r)))
			pilesForOthers: set[Pile] = set(range(len(folding) - state.pile - 次)).intersection(domain_r)
			for pileSacrifice in pilesForOthers:
				state.boxOfPermutationSpace = 是valid(atPileExcludeLeaf_inBulk(state.boxOfPermutationSpace, pileSacrifice, r))

			domain_k: Iterable[Pile] = tuple(filter(state.pile.__gt__, getDomainLeaf(state, k)))

			state = excludeLeaf_rBeforeLeaf_k(state, k, r, domain_k, set(domain_r).difference(pilesForOthers))

		state.removeCreaseViolations().reduceAllPermutationSpace()

	state.boxOfPermutationSpace.extend(boxOfPermutationSpace)
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
