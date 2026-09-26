from __future__ import annotations

from functools import reduce
from mapFolding.algorithms.matrixMeanders import walkDyckPath
from mapFolding.algorithms.matrixMeandersShare import makeLookupMeanders, shortcut
from mapFolding.dataBaskets import StateMeanders
from research.matrixMeanders.formulasTriangle import A005315of0, boxOfDiagonals

def count(state: StateMeanders) -> StateMeanders:
	while 0 < state.boundary:
		def analyzeArcCode(arcCode: int, meanders: int, state: StateMeanders = state) -> None:
			bitsAlfa: int = arcCode & state.bitsLocator
			bitsAlfaHasArcs: bool = 1 < bitsAlfa
			bitsAlfaIsEven: int = bitsAlfa & 1 ^ 1

			bitsZulu: int = arcCode >> 1 & state.bitsLocator
			bitsZuluHasArcs: bool = 1 < bitsZulu
			bitsZuluIsEven: int = bitsZulu & 1 ^ 1

			arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlfa) << 2 | 3  # No stack required. Evaluate formula step-wise left to right: (parentheses) override precedence.
			if arcCodeAnalysis < state.arcCodeMAXIMUM:
				state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			if bitsAlfaHasArcs:
				arcCodeAnalysis = bitsAlfaIsEven << 1 | bitsAlfa >> 2 | bitsZulu << 3  # Stack of 1: `bitsAlfaIsEven` has `bitsAlfa` in it.
				if arcCodeAnalysis < state.arcCodeMAXIMUM:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			if bitsZuluHasArcs:
				arcCodeAnalysis = bitsZuluIsEven | bitsAlfa << 2 | bitsZulu >> 1  # Stack of 1: `bitsZuluIsEven` has `bitsZulu` in it.
				if arcCodeAnalysis < state.arcCodeMAXIMUM:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			if bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):  # Stack of 1 if repeat bitsLocator operations. Lots of stacks to avoid duplicate work.
				# This analysis might modify `bitsAlfa` or `bitsZulu`, so it should be last.
				if bitsAlfaIsEven and not bitsZuluIsEven:
					bitsAlfa ^= walkDyckPath(bitsAlfa)
				elif bitsZuluIsEven and not bitsAlfaIsEven:
					bitsZulu ^= walkDyckPath(bitsZulu)

				arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2  # No stack required. Evaluate formula step-wise left to right: (parentheses) override precedence.
				if arcCodeAnalysis < state.arcCodeMAXIMUM:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

		state.reduceBoundary()

		lookupArcCodeMeanders: dict[int, int] = state.lookupMeanders.copy()
		state.lookupMeanders = {}

		tuple(map(analyzeArcCode, lookupArcCodeMeanders.keys(), lookupArcCodeMeanders.values()))

		state = shortcut(state)

	return state

def doTheNeedful(state: StateMeanders) -> int:
	state.n = 0
	return sum(count(state).lookupMeanders.values()) + state.n

def countDiagonal(n: int, 次diagonal: int) -> int:
	arcCode: int = (1 << (2 * (n - 2 * 次diagonal) + 2)) - 1
	state: StateMeanders = StateMeanders(n, 'semi', lookupMeanders={arcCode: 1})
	return doTheNeedful(state)

def countSemiMeandersWithPruning(n: int) -> int:
	diagonals: int = len(boxOfDiagonals)
	totalDiagonals: int = n // 2
	countTotal: int = 0
	if diagonals < totalDiagonals:
		lookupMeanders: dict[int, int] = dict(tuple(makeLookupMeanders('semi', n).items())[:-diagonals])
		state: StateMeanders = StateMeanders(n, 'semi', lookupMeanders=lookupMeanders)
		countTotal = doTheNeedful(state)

	if n <= 2:
		countTotal = A005315of0(n)
	else:
		countTotal += reduce(lambda subtotal, diagonal: subtotal + diagonal(n), boxOfDiagonals[0:totalDiagonals], 0)

	return countTotal
