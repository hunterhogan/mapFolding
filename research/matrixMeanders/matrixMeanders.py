# ruff: ignore[undocumented-public-module]
# TODO Generate this module with kitAST
# ruff: file-ignore[function-uses-loop-variable]
# ruff: file-ignore[undocumented-public-function]
from __future__ import annotations

from functools import cache
from research.matrixMeanders.matrixMeandersPrune import shortcut
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeanders

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
	findTheExtra_0b1: int = 0
	flipExtra_0b1_Here: int = 1
	while 0 <= findTheExtra_0b1:
		flipExtra_0b1_Here <<= 2
		if intWithExtra_0b1 & flipExtra_0b1_Here == 0:
			findTheExtra_0b1 += 1
		else:
			findTheExtra_0b1 -= 1
	return flipExtra_0b1_Here

def count(state: StateMeanders) -> StateMeanders:
	while 0 < state.boundary:
		def analyzeArcCode(arcCode: int, meanders: int) -> None:
			bitsAlfa: int = arcCode & state.bitsLocator
			bitsAlfaHasArcs: bool = 1 < bitsAlfa
			bitsAlfaIsEven: int = bitsAlfa & 1 ^ 1

			bitsZulu: int = arcCode >> 1 & state.bitsLocator
			bitsZuluHasArcs: bool = 1 < bitsZulu
			bitsZuluIsEven: int = bitsZulu & 1 ^ 1

			arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlfa) << 2 | 3  # Evaluate formula step-wise left to right: (parentheses) override precedence.
			if arcCodeAnalysis < state.arcCodeMAXIMUM:
				state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			if bitsAlfaHasArcs:
				arcCodeAnalysis = bitsAlfaIsEven << 1 | bitsAlfa >> 2 | bitsZulu << 3  # `bitsAlfaIsEven` has `bitsAlfa` in it.
				if arcCodeAnalysis < state.arcCodeMAXIMUM:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			if bitsZuluHasArcs:
				arcCodeAnalysis = bitsZuluIsEven | bitsAlfa << 2 | bitsZulu >> 1  # `bitsZuluIsEven` has `bitsZulu` in it.
				if arcCodeAnalysis < state.arcCodeMAXIMUM:
					state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders

			if bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):
				# This analysis might modify `bitsAlfa` or `bitsZulu`, so it should be last.
				if bitsAlfaIsEven and not bitsZuluIsEven:
					bitsAlfa ^= walkDyckPath(bitsAlfa)
				elif bitsZuluIsEven and not bitsAlfaIsEven:
					bitsZulu ^= walkDyckPath(bitsZulu)

				arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2  # Evaluate formula step-wise left to right: (parentheses) override precedence.
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
