from __future__ import annotations

from mapFolding.algorithms.matrixAxisStrip import isJoined吗, makeLookupPartner, walkDyckPathAlfa, walkDyckPathZulu
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.dataBaskets import StateMeandersAxis

def reflectArcCode(arcCode: int) -> int:
	terminals: int = (arcCode.bit_length() >> 1) - 1
	bitGap: int = arcCode & (1 << terminals + 1) - 1
	arcCode >>= terminals + 1
	return (int(format(arcCode, 'b')[:0:-1], 2) ^ (1 << terminals + 1) - 1) << terminals + 1 | (1 << terminals) // bitGap

def count(state: StateMeandersAxis) -> StateMeandersAxis:
	while 0 < state.boundary:
		def analyzeArcCode(arcCode: int, meanders: int, state: StateMeandersAxis = state) -> None:
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

def isHalfJoined吗(arcCode: int) -> bool:
	terminals: int = (arcCode.bit_length() >> 1) - 1
	return 0 < arcCode & 0b11 << (terminals >> 1) and isJoined吗(makeLookupPartner(arcCode >> terminals + 1))

def doTheNeedful(state: StateMeandersAxis) -> StateMeandersAxis:
	state = count(state)
	state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add
		, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
	state.lookupMeanders = {}
	return state
