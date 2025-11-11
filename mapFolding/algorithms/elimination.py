from cytoolz.functoolz import curry as syntacticCurry
from cytoolz.itertoolz import groupby as toolz_groupby
from itertools import pairwise, repeat
from mapFolding.algorithms.eliminationCount import count, permutands
from mapFolding.algorithms.iff import productOfDimensions
from mapFolding.algorithms.pinning2Dn import pinByFormula
from mapFolding.dataBaskets import EliminationState
from math import factorial
from more_itertools import flatten, iter_index, unique
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable

@syntacticCurry
def _excludeLeafAtColumn(sequencePinnedLeaves: list[dict[int, int]], leaf: int, column: int, leavesTotal: int) -> list[dict[int, int]]:
	listPinnedLeaves: list[dict[int, int]] = []
	for pinnedLeaves in sequencePinnedLeaves:
		leafAtColumn: int | None = pinnedLeaves.get(column)
		if leaf == leafAtColumn:
			continue																						# Exclude `leaf` previously fixed at `column`.
		if (leafAtColumn is not None) or (leaf in pinnedLeaves.values()):									# `column` is occupied, which excludes `leaf`.
			listPinnedLeaves.append(pinnedLeaves)															# Or `leaf` is pinned, but not at `column`.
			continue
		deconstructedPinnedLeaves: dict[int, dict[int, int]] = deconstructPinnedLeaves(pinnedLeaves, column, leavesTotal)
		deconstructedPinnedLeaves.pop(leaf)																	# Exclude dictionary with `leaf` fixed at `column`.
		listPinnedLeaves.extend(deconstructedPinnedLeaves.values())
	return listPinnedLeaves

def pinLeafAtColumn(sequencePinnedLeaves: list[dict[int, int]], leaf: int, column: int, leavesTotal: int) -> list[dict[int, int]]:
	listPinnedLeaves: list[dict[int, int]] = []
	for pinnedLeaves in sequencePinnedLeaves:
		leafAtColumn: int | None = pinnedLeaves.get(column)
		if leaf == leafAtColumn:
			listPinnedLeaves.append(pinnedLeaves)															# That was easy.
			continue
		if leafAtColumn is not None:																		# `column` is occupied, but not by `leaf`, so exclude it.
			continue
		listPinnedLeaves.append(deconstructPinnedLeaves(pinnedLeaves, column, leavesTotal).pop(leaf))		# Keep the dictionary with `leaf` fixed at `column`.
	return listPinnedLeaves

@syntacticCurry
def _isPinnedAtColumn(pinnedLeaves: dict[int, int], leaf: int, column: int) -> bool:
	return leaf == pinnedLeaves.get(column)

def _segregatePinnedAtColumn(listPinnedLeaves: list[dict[int, int]], leaf: int, column: int) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
	isPinned: Callable[[dict[int, int]], bool] = _isPinnedAtColumn(leaf=leaf, column=column)
	grouped: dict[bool, list[dict[int, int]]] = toolz_groupby(isPinned, listPinnedLeaves)
	return (grouped.get(False, []), grouped.get(True, []))

@syntacticCurry
def atColumnPinLeaf(pinnedLeaves: dict[int, int], column: int, leaf: int) -> dict[int, int]:
	dictionaryPinnedLeaves: dict[int, int] = dict(pinnedLeaves)
	dictionaryPinnedLeaves[column] = leaf
	return dictionaryPinnedLeaves

def deconstructPinnedLeaves(pinnedLeaves: dict[int, int], column: int, leavesTotal: int) -> dict[int, dict[int, int]]:
	"""Replace `pinnedLeaves`, which doesn't pin a leaf at `column`, with the equivalent group of dictionaries, which each pin a distinct leaf at `column`.

	Parameters
	----------
	pinnedLeaves : dict[int, int]
		Dictionary to divide and replace.
	column : int
		Column in which to pin a leaf.

	Returns
	-------
	deconstructedPinnedLeaves : dict[int, dict[int, int]]
		Dictionary mapping from `leaf` pinned at `column` to the dictionary with the `leaf` pinned at `column`.
	"""
	leafAtColumn: int | None = pinnedLeaves.get(column)
	if leafAtColumn is not None:
		deconstructedPinnedLeaves: dict[int, dict[int, int]] = {leafAtColumn: pinnedLeaves}
	else:
		pin: Callable[[int], dict[int, int]] = atColumnPinLeaf(pinnedLeaves, column)
		deconstructedPinnedLeaves = {leaf: pin(leaf) for leaf in permutands(pinnedLeaves, leavesTotal)}
	return deconstructedPinnedLeaves

def DOTvalues[个](dictionary: dict[Any, 个]) -> list[个]:
	return list(dictionary.values())

def deconstructListPinnedLeaves(listPinnedLeaves: list[dict[int, int]], column: int, leavesTotal: int) -> list[dict[int, int]]:
	return list(flatten(map(DOTvalues, map(deconstructPinnedLeaves, listPinnedLeaves, repeat(column), repeat(leavesTotal)))))

def _excludeLeafRBeforeLeafK(state: EliminationState, k: int, r: int, columnK: int, listPinnedLeaves: list[dict[int, int]]) -> list[dict[int, int]]:
	listPinnedLeaves = deconstructListPinnedLeaves(listPinnedLeaves, columnK, state.leavesTotal)
	listPinned: list[dict[int, int]] = []
	for column in range(columnK, state.columnLast + 1):
		(listPinnedLeaves, listPinnedAtColumn) = _segregatePinnedAtColumn(listPinnedLeaves, k, column)
		listPinned.extend(listPinnedAtColumn)
	listPinnedLeaves.extend(_excludeLeafAtColumn(listPinned, r, columnK - 1, state.leavesTotal))
	return listPinnedLeaves

def excludeLeafRBeforeLeafK(state: EliminationState, k: int, r: int) -> EliminationState:
	for columnK in range(state.columnLast, 0, -1):
		state.listPinnedLeaves = _excludeLeafRBeforeLeafK(state, k, r, columnK, state.listPinnedLeaves)
	return state

def theorem4(state: EliminationState) -> EliminationState:
# ------- Lunnon Theorem 4: "G(p^d) is divisible by d!p^d." ---------------
	for listIndicesSameMagnitude in [list(iter_index(state.mapShape, magnitude)) for magnitude in unique(state.mapShape)]:
		if len(listIndicesSameMagnitude) > 1:
			state.subsetsTheorem4 = factorial(len(listIndicesSameMagnitude))
			for dimensionAlpha, dimensionBeta in pairwise(listIndicesSameMagnitude):
				k, r = (productOfDimensions(state.mapShape, dimension) + 1 for dimension in (dimensionAlpha, dimensionBeta))
				state = excludeLeafRBeforeLeafK(state, k, r)
	return state

def theorem2b(state: EliminationState) -> EliminationState:
# ------- Lunnon Theorem 2(b): "If some pᵢ > 2, G is divisible by 2n." -----------------------------
	if state.subsetsTheorem4 == 1 and max(state.mapShape) > 2:
		state.subsetsTheorem2 = 2
		dimension: int = state.mapShape.index(max(state.mapShape))
		k: int = productOfDimensions(state.mapShape, dimension) + 1
		r: int = 2 * k
		state = excludeLeafRBeforeLeafK(state, k, r)

	return state

def doTheNeedful(state: EliminationState, workersMaximum: int) -> EliminationState:  # noqa: ARG001
	"""Count the number of valid foldings for a given number of leaves."""
# ------- Lunnon Theorem 2(a): foldsTotal is divisible by leavesTotal; Pin leaf1 in column0 and exclude leaf2--leafN at column0 ----------------------------
	state.listPinnedLeaves = [{0: 1}]

	state = theorem4(state)
	state = theorem2b(state)
	state = pinByFormula(state)
	state = count(state)

	return state
