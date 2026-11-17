from __future__ import annotations

from mapFolding.algorithms.matrixAxisPile import getBitsLocator, pairTerminals
from mapFolding.algorithms.matrixMeanders import walkDyckPath
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from mapFolding.dataBaskets import StateMeandersAxis

def reflectArcCode(arcCode: int) -> int:
    if arcCode < 0:
        arcCode = ~arcCode
        bitStart: int = arcCode & (1 << (arcCode.bit_length() >> 1)) - 1
        arcCode >>= arcCode.bit_length() >> 1
        bitsLocator: int = getBitsLocator(arcCode)
        arcCode = (arcCode & bitsLocator) << 1 | arcCode >> 1 & bitsLocator
        return ~(arcCode << arcCode.bit_length() | (bitStart & bitsLocator) << 1 | bitStart >> 1 & bitsLocator)
    bitsLocator = getBitsLocator(arcCode)
    return (arcCode & bitsLocator) << 1 | arcCode >> 1 & bitsLocator

def endStrandAtDeadEnd(bitsAlfa: int, bitsZulu: int) -> int:
    if bitsAlfa & 1 == 0:
        bitStart: int = walkDyckPath(bitsAlfa) >> 2
        bitsAlfa >>= 2
        bitsAlfa = bitsAlfa & bitStart - 1 | bitsAlfa >> 2 & -bitStart
    else:
        bitStart = walkDyckPath(bitsZulu << 2) >> 2
        bitsAlfa >>= 2
        bitsZulu = bitsZulu & bitStart - 1 | bitsZulu >> 2 & -bitStart
        bitStart <<= 1
    arcCode: int = bitsZulu << 1 | bitsAlfa
    return ~(arcCode << arcCode.bit_length() | bitStart)

def count(state: StateMeandersAxis) -> StateMeandersAxis:
    while 0 < state.boundary:

        def analyzeArcCode(arcCode: int, meanders: int, state: StateMeandersAxis=state) -> None:
            if 0 <= arcCode:
                bitsAlfa: int = arcCode & state.bitsLocator
                bitsAlfaHasArcs: bool = 1 < bitsAlfa
                bitsAlfaIsEven: int = bitsAlfa & 1 ^ 1
                bitsZulu: int = arcCode >> 1 & state.bitsLocator
                bitsZuluHasArcs: bool = 1 < bitsZulu
                bitsZuluIsEven: int = bitsZulu & 1 ^ 1
                arcCodeAnalysis: int = (bitsZulu << 1 | bitsAlfa) << 2 | 3
                if arcCodeAnalysis < state.arcCodeMaximum:
                    arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if bitsAlfaHasArcs:
                    arcCodeAnalysis = bitsAlfaIsEven << 1 | bitsAlfa >> 2 | bitsZulu << 3
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if bitsZuluHasArcs:
                    arcCodeAnalysis = bitsZuluIsEven | bitsAlfa << 2 | bitsZulu >> 1
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if not bitsZuluHasArcs or state.kind == 'pileDeadEnd':
                    arcCodeAnalysis = ~(arcCode << arcCode.bit_length() | 1)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                    if bitsAlfaHasArcs:
                        arcCodeAnalysis = endStrandAtDeadEnd(bitsAlfa, bitsZulu)
                        if arcCodeAnalysis < state.arcCodeMaximum:
                            arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                            state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if not bitsAlfaHasArcs or state.kind == 'pileDeadEnd':
                    arcCodeAnalysis = ~(arcCode << arcCode.bit_length() | 2)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                    if bitsZuluHasArcs:
                        arcCodeAnalysis = reflectArcCode(endStrandAtDeadEnd(bitsZulu, bitsAlfa))
                        if arcCodeAnalysis < state.arcCodeMaximum:
                            arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                            state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):
                    if bitsAlfaIsEven and (not bitsZuluIsEven):
                        bitsAlfa ^= walkDyckPath(bitsAlfa)
                    elif bitsZuluIsEven and (not bitsAlfaIsEven):
                        bitsZulu ^= walkDyckPath(bitsZulu)
                    arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
            else:
                arcCode = ~arcCode
                bitStart: int = arcCode & (1 << (arcCode.bit_length() >> 1)) - 1
                arcCode >>= arcCode.bit_length() >> 1
                bitsAlfa = arcCode & state.bitsLocator
                bitsAlfaHasArcs = 1 < bitsAlfa
                bitsAlfaIsEven = bitsAlfa & 1 ^ 1
                bitStartAlfa: int = bitStart & state.bitsLocator
                bitsZulu = arcCode >> 1 & state.bitsLocator
                bitsZuluHasArcs = 1 < bitsZulu
                bitsZuluIsEven = bitsZulu & 1 ^ 1
                bitStartZulu: int = bitStart >> 1 & state.bitsLocator
                arcCodeAnalysis = (bitsZulu << 1 | bitsAlfa) << 2 | 3
                arcCodeAnalysis = ~(arcCodeAnalysis << arcCodeAnalysis.bit_length() | bitStart << 2)
                if arcCodeAnalysis < state.arcCodeMaximum:
                    arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if bitStart == 1:
                    arcCodeAnalysis = ~(arcCode << arcCode.bit_length() | 2)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                elif bitsAlfaHasArcs:
                    arcCodeAnalysis = bitsAlfaIsEven << 1 | bitsAlfa >> 2 | bitsZulu << 3
                    arcCodeAnalysis = ~(arcCodeAnalysis << arcCodeAnalysis.bit_length() | bitStartAlfa >> 2 | bitStartZulu << 3)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if bitStart == 2:
                    arcCodeAnalysis = ~(arcCode << arcCode.bit_length() | 1)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                elif bitsZuluHasArcs:
                    arcCodeAnalysis = bitsZuluIsEven | bitsAlfa << 2 | bitsZulu >> 1
                    arcCodeAnalysis = ~(arcCodeAnalysis << arcCodeAnalysis.bit_length() | bitStartZulu >> 1 | bitStartAlfa << 2)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                if bitStart == 1 and bitsZuluHasArcs:
                    arcCodeAnalysis = reflectArcCode(endStrandAtDeadEnd(bitsZulu, bitsAlfa))
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                elif bitStart == 2 and bitsAlfaHasArcs:
                    arcCodeAnalysis = endStrandAtDeadEnd(bitsAlfa, bitsZulu)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
                elif bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):
                    if bitsAlfaIsEven and (not bitsZuluIsEven):
                        bitsAlfa ^= walkDyckPath(bitsAlfa)
                    elif bitsZuluIsEven and (not bitsAlfaIsEven):
                        bitsZulu ^= walkDyckPath(bitsZulu)
                    arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2
                    arcCodeAnalysis = ~(arcCodeAnalysis << arcCodeAnalysis.bit_length() | bitStart >> 2)
                    if arcCodeAnalysis < state.arcCodeMaximum:
                        arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                        state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
        state.reduceBoundary()
        lookupArcCodeMeanders: dict[int, int] = state.lookupMeanders.copy()
        state.lookupMeanders = {}
        tuple(map(analyzeArcCode, lookupArcCodeMeanders.keys(), lookupArcCodeMeanders.values(), strict=True))
    return state

def makeLookupPartner(arcCode: int, terminal: int) -> list[int]:
    bitsLocator: int = getBitsLocator(arcCode)
    bitsAlfa: int = arcCode & bitsLocator
    bitsZulu: int = arcCode >> 1 & bitsLocator
    lookupPartner: list[int] = list(range((bitsAlfa.bit_length() + bitsZulu.bit_length()) // 2))
    boxOfCrossAlfa: list[int] = pairTerminals(bitsAlfa, filter(terminal.__ne__, range(len(lookupPartner))), lookupPartner)
    boxOfCrossZulu: list[int] = pairTerminals(bitsZulu, filter(terminal.__ne__, range(len(lookupPartner) - 1, -1, -1)), lookupPartner)
    tuple(map(lookupPartner.__setitem__, boxOfCrossAlfa, boxOfCrossZulu, strict=True))
    tuple(map(lookupPartner.__setitem__, boxOfCrossZulu, boxOfCrossAlfa, strict=True))
    return lookupPartner

def isJoined吗(lookupPartner: Sequence[int], terminal: int) -> bool:
    strands: int = 1
    while terminal != 0:
        terminal = lookupPartner[len(lookupPartner) - terminal]
        strands += 1
    return 2 * strands - 1 == len(lookupPartner)

def isHalfJoined吗(arcCode: int) -> bool:
    if 0 <= arcCode:
        return False
    arcCode = ~arcCode
    bitStart: int = arcCode & (1 << (arcCode.bit_length() >> 1)) - 1
    arcCode >>= arcCode.bit_length() >> 1
    bitsLocator: int = getBitsLocator(arcCode)
    terminalsAlfa: int = (arcCode & bitsLocator).bit_length() >> 1
    terminalsZulu: int = (arcCode >> 1 & bitsLocator).bit_length() >> 1
    terminal: int = bitStart.bit_length() >> 1
    if bitStart & bitsLocator:
        terminalsAlfa += 1
    else:
        terminalsZulu += 1
        terminal = terminalsAlfa + terminalsZulu - terminal
    return terminalsAlfa == terminalsZulu + 1 and isJoined吗(makeLookupPartner(arcCode, terminal), terminal)

def doTheNeedful(state: StateMeandersAxis) -> StateMeandersAxis:
    state = count(state)
    state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
    state.lookupMeanders = {}
    return state
