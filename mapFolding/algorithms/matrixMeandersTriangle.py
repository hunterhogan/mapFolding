from __future__ import annotations

from mapFolding.algorithms.matrixMeanders import walkDyckPath
from mapFolding.algorithms.matrixMeandersShare import shortcut
from mapFolding.dataBaskets import StateMeanders

def count(state: StateMeanders) -> StateMeanders:
	state = shortcut(state)
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

def doTheNeedful(state: StateMeanders) -> StateMeanders:
	return count(state)

def countDiagonal(n: int, 次diagonal: int) -> StateMeanders:
	arcCode: int = (1 << (2 * (n - 2 * 次diagonal) + 2)) - 1
	return StateMeanders(n, 'semi', lookupMeanders={arcCode: 1})
