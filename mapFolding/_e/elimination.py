from cytoolz.functoolz import curry as syntacticCurry
from cytoolz.itertoolz import groupby as toolz_groupby
from itertools import pairwise, repeat
from mapFolding._e import decreasing, fullRange
from mapFolding._e.eliminationCount import count, permutands
from mapFolding._e.pinning2Dn import pinByFormula
from mapFolding.algorithms.iff import productOfDimensions
from mapFolding.dataBaskets import EliminationState
from math import factorial
from more_itertools import flatten, iter_index, unique
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

@syntacticCurry
def _excludeLeafAtPile(sequencePinnedLeaves: list[dict[int, int]], leaf: int, pile: int, leavesTotal: int) -> list[dict[int, int]]:
	listPinnedLeaves: list[dict[int, int]] = []
	for pinnedLeaves in sequencePinnedLeaves:
		leafAtPile: int | None = pinnedLeaves.get(pile)
		if leaf == leafAtPile:
			continue																					# Exclude `leaf` previously fixed at `pile`.
		if (leafAtPile is not None) or (leaf in pinnedLeaves.values()):									# `pile` is occupied, which excludes `leaf`.
			listPinnedLeaves.append(pinnedLeaves)														# Or `leaf` is pinned, but not at `pile`.
			continue
		deconstructedPinnedLeaves: dict[int, dict[int, int]] = deconstructPinnedLeaves(pinnedLeaves, pile, leavesTotal)
		deconstructedPinnedLeaves.pop(leaf)																# Exclude dictionary with `leaf` fixed at `pile`.
		listPinnedLeaves.extend(deconstructedPinnedLeaves.values())
	return listPinnedLeaves

def pinLeafAtPile(sequencePinnedLeaves: list[dict[int, int]], leaf: int, pile: int, leavesTotal: int) -> list[dict[int, int]]:
	listPinnedLeaves: list[dict[int, int]] = []
	for pinnedLeaves in sequencePinnedLeaves:
		leafAtPile: int | None = pinnedLeaves.get(pile)
		if leaf == leafAtPile:
			listPinnedLeaves.append(pinnedLeaves)														# That was easy.
			continue
		if leafAtPile is not None:																		# `pile` is occupied, but not by `leaf`, so exclude it.
			continue
		listPinnedLeaves.append(deconstructPinnedLeaves(pinnedLeaves, pile, leavesTotal).pop(leaf))		# Keep the dictionary with `leaf` fixed at `pile`.
	return listPinnedLeaves

@syntacticCurry
def _isPinnedAtPile(pinnedLeaves: dict[int, int], leaf: int, pile: int) -> bool:
	return leaf == pinnedLeaves.get(pile)

def _segregatePinnedAtPile(listPinnedLeaves: list[dict[int, int]], leaf: int, pile: int) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
	isPinned: Callable[[dict[int, int]], bool] = _isPinnedAtPile(leaf=leaf, pile=pile)
	grouped: dict[bool, list[dict[int, int]]] = toolz_groupby(isPinned, listPinnedLeaves)
	return (grouped.get(False, []), grouped.get(True, []))

@syntacticCurry
def atPilePinLeaf(pinnedLeaves: dict[int, int], pile: int, leaf: int) -> dict[int, int]:
	dictionaryPinnedLeaves: dict[int, int] = dict(pinnedLeaves)
	dictionaryPinnedLeaves[pile] = leaf
	return dictionaryPinnedLeaves

def deconstructPinnedLeaves(pinnedLeaves: dict[int, int], pile: int, leavesTotal: int) -> dict[int, dict[int, int]]:
	"""Replace `pinnedLeaves`, which doesn't pin a leaf at `pile`, with the equivalent group of dictionaries, which each pin a distinct leaf at `pile`.

	Parameters
	----------
	pinnedLeaves : dict[int, int]
		Dictionary to divide and replace.
	pile : int
		Pile in which to pin a leaf.

	Returns
	-------
	deconstructedPinnedLeaves : dict[int, dict[int, int]]
		Dictionary mapping from `leaf` pinned at `pile` to the dictionary with the `leaf` pinned at `pile`.
	"""
	leafAtPile: int | None = pinnedLeaves.get(pile)
	if leafAtPile is not None:
		deconstructedPinnedLeaves: dict[int, dict[int, int]] = {leafAtPile: pinnedLeaves}
	else:
		pin: Callable[[int], dict[int, int]] = atPilePinLeaf(pinnedLeaves, pile)
		deconstructedPinnedLeaves = {leaf: pin(leaf) for leaf in permutands(pinnedLeaves, leavesTotal)}
	return deconstructedPinnedLeaves

def DOTvalues[个](dictionary: dict[Any, 个]) -> list[个]:
	return list(dictionary.values())

def deconstructListPinnedLeaves(listPinnedLeaves: list[dict[int, int]], pile: int, leavesTotal: int) -> list[dict[int, int]]:
	return list(flatten(map(DOTvalues, map(deconstructPinnedLeaves, listPinnedLeaves, repeat(pile), repeat(leavesTotal)))))

def _excludeLeafRBeforeLeafK(state: EliminationState, k: int, r: int, pileK: int, listPinnedLeaves: list[dict[int, int]]) -> list[dict[int, int]]:
	listPinnedLeaves = deconstructListPinnedLeaves(listPinnedLeaves, pileK, state.leavesTotal)
	listPinned: list[dict[int, int]] = []
	for pile in range(pileK, state.pileLast + fullRange):
		(listPinnedLeaves, listPinnedAtPile) = _segregatePinnedAtPile(listPinnedLeaves, k, pile)
		listPinned.extend(listPinnedAtPile)
	listPinnedLeaves.extend(_excludeLeafAtPile(listPinned, r, pileK - 1, state.leavesTotal))
	return listPinnedLeaves

def excludeLeafRBeforeLeafK(state: EliminationState, k: int, r: int) -> EliminationState:
	for pileK in range(state.pileLast, 0, decreasing):
		state.listPinnedLeaves = _excludeLeafRBeforeLeafK(state, k, r, pileK, state.listPinnedLeaves)
	return state

def theorem4(state: EliminationState) -> EliminationState:
# ------- Lunnon Theorem 4: "G(p^d) is divisible by d!p^d." ---------------
	for listIndicesSameMagnitude in [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]:
		if len(listIndicesSameMagnitude) > 1:
			state.subsetsTheorem4 = factorial(len(listIndicesSameMagnitude))
			for dimensionAlpha, dimensionBeta in pairwise(listIndicesSameMagnitude):
				k, r = (productOfDimensions(state.mapShape, dimension) for dimension in (dimensionAlpha, dimensionBeta))
				state = excludeLeafRBeforeLeafK(state, k, r)
	return state

def theorem2b(state: EliminationState) -> EliminationState:
# ------- Lunnon Theorem 2(b): "If some pᵢ > 2, G is divisible by 2n." -----------------------------
	if state.subsetsTheorem4 == 1 and max(state.mapShape) > 2 and (state.leavesTotal > 4):
		state.subsetsTheorem2 = 2
		dimension: int = state.mapShape.index(max(state.mapShape))
		k: int = productOfDimensions(state.mapShape, dimension)
		r: int = 2 * k
		state = excludeLeafRBeforeLeafK(state, k, r)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:  # noqa: ARG001
	"""Count the number of valid foldings for a given number of leaves."""
# ------- Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal; Pin leaf0 in pile0 and exclude leaf1--leafN-1 at pile0 ----------------------------
	state.listPinnedLeaves = [{0: 0}]

	state = theorem4(state)
	state = theorem2b(state)
	state = pinByFormula(state)
	state = count(state)

	return state  # noqa: RET504
