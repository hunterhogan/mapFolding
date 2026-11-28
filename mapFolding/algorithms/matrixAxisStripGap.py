"""Count symmetric meander shapes by cutting across the strip at the axis and marking the road face.

(AI generated docstring)

You can use this module to compute A223096(m), the number of symmetric meander shapes on 2m + 1
crossings. The transfer is the strip-cut transfer of `mapFolding.algorithms.matrixAxisStrip` with one
more mark. Each state is `arcCode = word << (N + 1) | 1 << g`: `word` is the circle word of
`mapFolding.algorithms.matrixAxisStrip` with its sentinel bit at position N, and the one-bit gap mark
`bitGap = 1 << g` records the face of the cut where the road leaves toward infinity. A state has
2N + 2 bits, so N is recoverable from `arcCode`.

Contents
--------
Functions
	count
		Advance the strip-cut transfer with the gap mark to the symmetry axis.
	doTheNeedful
		Count the halves that join at the symmetry axis and add the count to `state.countAddend`.
	isHalfJoined吗
		Decide whether the half that a marked circle word describes joins with its half-turn image.
	reflectArcCode
		Reflect a marked circle word across the strip.
"""
from __future__ import annotations

from mapFolding.algorithms.matrixAxisStrip import isJoined吗, makeLookupPartner, walkDyckPathAlfa, walkDyckPathZulu
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeandersAxis

# TODO remake `reflectArcCode` from scratch.
def reflectArcCode(arcCode: int) -> int:
	"""Reflect a marked circle word across the strip.

	(AI generated docstring)

	The reflection σ reflects the circle word as
	`mapFolding.algorithms.matrixAxisStrip.reflectArcCode` does and moves the gap mark from
	position g to position N − g.

	Parameters
	----------
	arcCode : int
		Marked circle word, `word << (N + 1) | 1 << g`.

	Returns
	-------
	arcCodeReflected : int
		Marked circle word of the reflected half.
	"""
	terminals: int = (arcCode.bit_length() >> 1) - 1
	bitGap: int = arcCode & (1 << terminals + 1) - 1
	arcCode >>= terminals + 1
	return (int(format(arcCode, 'b')[:0:-1], 2) ^ (1 << terminals + 1) - 1) << terminals + 1 | (1 << terminals) // bitGap

def count(state: StateMeandersAxis) -> StateMeandersAxis:
	"""Advance the strip-cut transfer with the gap mark to the symmetry axis.

	(AI generated docstring)

	You can use this function to obtain, for each marked circle word at the symmetry axis, the number
	of halves that end with that word. The four moves are the moves of
	`mapFolding.algorithms.matrixAxisStrip.count`, and the gap mark shifts with the word. A move
	cannot remove the terminal at the gap: the move that brings in the terminal of bank 0 requires g ≠
	0, the move that brings in the terminal of bank 1 requires g ≠ N, and the move that brings in both
	terminals requires both conditions.

	Parameters
	----------
	state : StateMeandersAxis
		Transfer state of kind `'stripGap'`. `state.lookupMeanders` maps each marked circle word to
		its count.

	Returns
	-------
	state : StateMeandersAxis
		The same state. `state.lookupMeanders` holds the counts at the symmetry axis.
	"""
	while 0 < state.boundary:
		def analyzeArcCode(arcCode: int, meanders: int, state: StateMeandersAxis = state) -> None:
			"""Accumulate one marked circle word's successors while preserving the road gap.

			(AI generated docstring)

			Input Mutation
			--------------
			`state.lookupMeanders` receives `meanders` for each allowed successor below
			`state.arcCodeMaximum`. Moves that remove a terminal at the gap are excluded.

			Parameters
			----------
			arcCode : int
				Packed circle word and one-bit gap mark for the strip cut.
			meanders : int
				Number of halves represented by `arcCode`.
			state : StateMeandersAxis
				Enclosing marked strip-cut state by default, with the boundary already reduced.
			"""
			terminals: int = (arcCode.bit_length() >> 1) - 1
			bitGap: int = arcCode & (1 << terminals + 1) - 1
			arcCode >>= terminals + 1

			arcCodeAnalysis: int = (arcCode << 1 | 1 << terminals + 2) << terminals + 3 | bitGap << 1
			if arcCodeAnalysis < state.arcCodeMaximum:
				state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			flipAlfa: int = walkDyckPathAlfa(arcCode)
			if 1 < bitGap:
				arcCodeAnalysis = (arcCode ^ flipAlfa | 1 << terminals + 1) >> 1 << terminals + 1 | bitGap >> 1
				if arcCodeAnalysis < state.arcCodeMaximum:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			flipZulu: int = walkDyckPathZulu(arcCode)
			if bitGap < 1 << terminals:
				arcCodeAnalysis = (arcCode ^ flipZulu ^ 1 << terminals) << terminals + 2 | bitGap << 1
				if arcCodeAnalysis < state.arcCodeMaximum:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			if 1 < flipZulu and 1 < bitGap < 1 << terminals:
				arcCodeAnalysis = ((arcCode ^ flipAlfa ^ flipZulu) >> 1 ^ 1 << terminals - 1) << terminals - 1 | bitGap >> 1
				if arcCodeAnalysis < state.arcCodeMaximum:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

		state.reduceBoundary()

		lookupArcCodeMeanders: dict[int, int] = state.lookupMeanders.copy()
		state.lookupMeanders = {}

		tuple(map(analyzeArcCode, lookupArcCodeMeanders.keys(), lookupArcCodeMeanders.values(), strict=True))

	return state

# TODO remake `isHalfJoined吗` from scratch.
def isHalfJoined吗(arcCode: int) -> bool:
	"""Decide whether the half that a marked circle word describes joins with its half-turn image.

	(AI generated docstring)

	The half joins when the gap mark is at position N / 2 or N / 2 + 1 and the circle word passes
	`mapFolding.algorithms.matrixAxisStrip.isJoined吗`.

	Parameters
	----------
	arcCode : int
		Marked circle word at the symmetry axis.

	Returns
	-------
	isHalfJoined : bool
		`True` when the half and its half-turn image form one symmetric meander shape.
	"""
	terminals: int = (arcCode.bit_length() >> 1) - 1
	return 0 < arcCode & 0b11 << (terminals >> 1) and isJoined吗(makeLookupPartner(arcCode >> terminals + 1))

def doTheNeedful(state: StateMeandersAxis) -> StateMeandersAxis:
	"""Count the halves that join at the symmetry axis and add the count to `state.countAddend`.

	(AI generated docstring)

	You can use this function to complete a term of A223096. With D(w) the count of marked circle word
	w at the symmetry axis, J the test `isHalfJoined吗`, and σ the reflection `reflectArcCode`, the
	term is Σ_w D(w)·(J(w) + J(σw)) / 2.

	Parameters
	----------
	state : StateMeandersAxis
		Transfer state of kind `'stripGap'` before the transfer.

	Returns
	-------
	state : StateMeandersAxis
		The same state with the term in `state.countAddend` and an empty `state.lookupMeanders`. Read
		the term from `state.totalMeanders`.
	"""
	state = count(state)
	state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add
		, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
	state.lookupMeanders = {}
	return state
