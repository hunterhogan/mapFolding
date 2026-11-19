from __future__ import annotations

from operator import add, mul, neg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from mapFolding.dataBaskets import StateMeandersAxis

def walkDyckPathAlfa(arcCode: int) -> int:
    findTheExtra_0b1: int = 0
    flipExtra_0b1_Here: int = 1
    while 0 <= findTheExtra_0b1:
        flipExtra_0b1_Here <<= 1
        if arcCode & flipExtra_0b1_Here == 0:
            findTheExtra_0b1 += 1
        else:
            findTheExtra_0b1 += neg(1)
    return flipExtra_0b1_Here

def walkDyckPathZulu(arcCode: int) -> int:
    findTheExtra_0b1: int = 0
    flipExtra_0b1_Here: int = 1 << arcCode.bit_length() - 2
    while 0 <= findTheExtra_0b1:
        flipExtra_0b1_Here >>= 1
        if arcCode & flipExtra_0b1_Here == 0:
            findTheExtra_0b1 += neg(1)
        else:
            findTheExtra_0b1 += 1
    return flipExtra_0b1_Here

def reflectArcCode(arcCode: int) -> int:
    return int(format(arcCode, 'b')[:0:-1], 2) ^ (1 << arcCode.bit_length()) - 1

def count(state: StateMeandersAxis) -> StateMeandersAxis:
    while 0 < state.boundary:

        def analyzeArcCode(arcCode: int, meanders: int, state: StateMeandersAxis=state) -> None:
            terminals: int = arcCode.bit_length() - 1
            arcCodeAnalysis: int = arcCode << 1 | 1 << terminals + 2
            if arcCodeAnalysis < state.arcCodeMaximum:
                arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            flipAlfa: int = walkDyckPathAlfa(arcCode)
            arcCodeAnalysis = (arcCode ^ flipAlfa | 1 << terminals + 1) >> 1
            if arcCodeAnalysis < state.arcCodeMaximum:
                arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            flipZulu: int = walkDyckPathZulu(arcCode)
            arcCodeAnalysis = (arcCode ^ flipZulu ^ 1 << terminals) << 1
            if arcCodeAnalysis < state.arcCodeMaximum:
                arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            if 1 < flipZulu:
                arcCodeAnalysis = (arcCode ^ flipAlfa ^ flipZulu) >> 1 ^ 1 << terminals - 1
                if arcCodeAnalysis < state.arcCodeMaximum:
                    arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
        state.reduceBoundary()
        lookupArcCodeMeanders: dict[int, int] = state.lookupMeanders.copy()
        state.lookupMeanders = {}
        tuple(map(analyzeArcCode, lookupArcCodeMeanders.keys(), lookupArcCodeMeanders.values(), strict=True))
    return state

def makeLookupPartner(arcCode: int) -> list[int]:
    lookupPartner: list[int] = list(range(arcCode.bit_length() - 1))
    boxOfOpen: list[int] = []
    terminal: int = 0
    while terminal < len(lookupPartner):
        if arcCode >> terminal & 1:
            lookupPartner[terminal] = boxOfOpen.pop()
            lookupPartner[lookupPartner[terminal]] = terminal
        else:
            boxOfOpen.append(terminal)
        terminal += 1
    return lookupPartner

def isJoined吗(lookupPartner: Sequence[int]) -> bool:
    terminal: int = lookupPartner[0]
    strands: int = 1
    while terminal != len(lookupPartner) // 2:
        terminal = lookupPartner[len(lookupPartner) - terminal]
        strands += 1
    return 2 * strands == len(lookupPartner)

def isHalfJoined吗(arcCode: int) -> bool:
    return isJoined吗(makeLookupPartner(arcCode))

def doTheNeedful(state: StateMeandersAxis) -> StateMeandersAxis:
    state = count(state)
    state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
    state.lookupMeanders = {}
    return state
