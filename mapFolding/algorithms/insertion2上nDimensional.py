"""Expand smaller-map albums into constrained higher-dimensional search spaces.

(AI generated docstring)

You can read an album for a map with one fewer dimension and build constrained elimination branches
for a map whose dimension lengths are all two. You can also record the folding collection already
present in state.

Contents
--------
Functions
	makeAlbum2上nDimensional
		Build target-map search spaces from an album with one fewer dimension.
	makeDescendants
		Build constrained target-map branches from one smaller-map folding.
	recordAlbum2上nDimensional
		Write the state's collected folding representatives to an album.
"""

# DEVELOPMENT module.
# pyright: reportUnusedVariable=false
from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from humpy_cytoolz import keyfilter as filterPile
from itertools import pairwise
from mapFolding import 一, 二, 零
from mapFolding.beDRY import makeAntiChoicesLeaf
from mapFolding.dataBaskets import PermutationSpace, StateElimination
from mapFolding.filters import 是valid
from mapFolding.kitFilesystem import makePathFilenameFolds, readAlbum, writeAlbum
from mapFolding.leafDomains import getDomainLeaf
from mapFolding.oeis import makeMapShape
from mapFolding.pinIt import atPileExcludeLeaf, excludeLeaf_rBeforeLeaf_k, pinLeavesDimension二, pinPilesAtEnds
from mapFolding.reduceIt import boxOfFunctionsReduction2上nDimensional, reduceLeafSpace
from mapFolding.theSSOT import settingsPackage
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
from Z0Z_tools import DOTitems

if TYPE_CHECKING:
	from collections.abc import Iterable
	from concurrent.futures import Future
	from mapFolding.theTypes import ChoicesLeaf, Folding, Leaf, Pile
	from pathlib import Path

def makeDescendants(folding: Folding, state: StateElimination, position: int) -> StateElimination:
	"""Build constrained target-map branches from one smaller-map folding.

	(AI generated docstring)

	You can use the leaf ordering in `folding` to restrict the target map in `state`. You try possible
	piles for the final leaf in `folding`, pin end piles, and reduce the resulting branches. You update
	`state` in place and receive progress output.

	Parameters
	----------
	folding : Folding
		Source folding from the album for a map with one fewer dimension.
	state : StateElimination
		Target elimination state with every dimension length equal to two and appropriate reductions
		configured.
	position : int
		Progress-display row assigned to this worker.

	Returns
	-------
	state : StateElimination
		The same state with accumulated constrained permutation spaces. You do not obtain a count of all
		descendants.
	"""
	leaf一: Leaf = folding[-1]
	domain一: Iterable[Pile] = getDomainLeaf(state, leaf一)
	leaves: tuple[Leaf, ...] = tuple(reversed(folding[2:-1]))
	antiChoicesLeaf: ChoicesLeaf = makeAntiChoicesLeaf(state.totalLeaves, leaves)
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
			if r in leaves二:
				leaves二.remove(r)
				if not leaves二:
					state = pinLeavesDimension二(state, CPUlimit=True)

			domain_r: Iterable[Pile] = tuple(filter(state.pile.__gt__, getDomainLeaf(state, r)))
			pilesForOthers: set[Pile] = set(range(len(folding) - state.pile - 次)).intersection(domain_r)
			for pileSacrifice in pilesForOthers:
				state.boxOfPermutationSpace = 是valid(atPileExcludeLeaf(state.boxOfPermutationSpace, pileSacrifice, r))

			domain_k: Iterable[Pile] = tuple(filter(state.pile.__gt__, getDomainLeaf(state, k)))

			state = excludeLeaf_rBeforeLeaf_k(state, k, r, domain_k, set(domain_r).difference(pilesForOthers))

		state.removeCreaseViolations().reduceAllPermutationSpace()

	state.boxOfPermutationSpace.extend(boxOfPermutationSpace)
	return state

def makeAlbum2上nDimensional(state: StateElimination, workersMaximum: int) -> StateElimination:
	"""Build target-map search spaces from an album with one fewer dimension.

	(AI generated docstring)

	You can read the previous-dimension album from `settingsPackage.pathAlbum` [1],
	process each source folding with `makeDescendants` [2], and gather worker permutation spaces and
	folding collections.

	Concurrency
	-----------
	Tasks run through `concurrent.futures.ProcessPoolExecutor` [3]. Each task receives a new elimination
	state initialized with a shallow copy of the supplied permutation-space list.

	Parameters
	----------
	state : StateElimination
		Target elimination state with every dimension length equal to two. The previous-dimension album must
		exist.
	workersMaximum : int
		Positive maximum number of worker processes.

	Returns
	-------
	state : StateElimination
		The same state with active spaces replaced by worker results and worker folding collections appended.

	Examples
	--------
	The insertion script prepares target-map search spaces from the preceding album.

		```python
		state = makeAlbum2上nDimensional(state, 14)
		```

	References
	----------
	[1] `mapFolding.theSSOT.settingsPackage.pathAlbum`

	[2] `makeDescendants`

	[3] Python concurrent.futures.ProcessPoolExecutor.
		https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
	"""
	album: Iterable[Folding] = readAlbum(makePathFilenameFolds(makeMapShape('A001417', state.totalDimensions - 1), settingsPackage.pathAlbum, suffix='.album'))

	with ProcessPoolExecutor(workersMaximum) as concurrencyManager:
		boxOfClaimTickets: list[Future[StateElimination]] = []
		for position, folding, in enumerate(album):
			stateCopy: StateElimination = StateElimination(state.mapShape, state.boxOfPermutationSpace.copy()
						, boxOfFunctionsReduction=boxOfFunctionsReduction2上nDimensional
					)
			boxOfClaimTickets.append(concurrencyManager.submit(makeDescendants, folding, stateCopy, position % workersMaximum + 1))

		state.boxOfPermutationSpace = []

		for claimTicket in tqdm(as_completed(boxOfClaimTickets), total=len(boxOfClaimTickets), disable=False, desc='for folding in album'):
			sherpa: StateElimination = claimTicket.result()
			state.boxOfPermutationSpace.extend(sherpa.boxOfPermutationSpace)
			state.boxOfFolding.extend(sherpa.boxOfFolding)

	return state

def recordAlbum2上nDimensional(state: StateElimination) -> Path:
	"""Write the state's collected folding representatives to an album.

	(AI generated docstring)

	You can sort `state.boxOfFolding` and save the collection with `mapFolding.kitFilesystem.writeAlbum`
	[1] to the album directory [2]. You save only collected foldings, without enumerating unresolved
	permutation spaces.

	Parameters
	----------
	state : StateElimination
		Elimination state providing the map shape and folding collection to write.

	Returns
	-------
	pathFilenameAlbum : Path
		Path of the album written for the state's map shape.

	Examples
	--------
	The insertion script records its collected foldings.

		```python
		recordAlbum2上nDimensional(state)
		```

	References
	----------
	[1] `mapFolding.kitFilesystem.writeAlbum`

	[2] `mapFolding.theSSOT.settingsPackage.pathAlbum`
	"""
	pathFilenameAlbum: Path = makePathFilenameFolds(state.mapShape, settingsPackage.pathAlbum, suffix='.album')
	writeAlbum(sorted(state.boxOfFolding), pathFilenameAlbum)
	return pathFilenameAlbum
