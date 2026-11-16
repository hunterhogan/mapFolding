from __future__ import annotations

from functools import cache
from mapFolding.algorithms.matrixAxisStrip import isJoined吗
from mapFolding.dataBaskets import StateMeanders
from operator import add, mul, neg
from typing import TYPE_CHECKING

@cache
def walkDyckPath(intWithExtra_0b1: int) -> int:
    """Locate the first Dyck-balance failure bit in `intWithExtra_0b1`.

    (AI generated docstring)

    You can use `walkDyckPath` to find the bit that must be toggled when an arc-joining transition in
    the meander transfer matrix closes a mismatched pair [1]. The `intWithExtra_0b1` value stores one
    side of the packed boundary state with parity bits at even positions.

    Parameters
    ----------
    intWithExtra_0b1 : int
        Packed bit pattern for one half of the current meander boundary state.

    Returns
    -------
    flipExtra_0b1_Here : int
        Bit mask `2^(2k)` at the first even-bit position where the prefix balance becomes negative.

    Bit Search
    ----------
    The scan advances by shifting `flipExtra_0b1_Here` left by `2` each step. The scan adds `1` when
    the bit is `0` and subtracts `1` when the bit is `1`. The function returns immediately at the
    first index where the running balance is negative in the Dyck-prefix sense.

    Mathematics
    -----------
    first negative prefix : equation
        ```text
        Let  x ≜ `intWithExtra_0b1`,  bᵢ ≜ bit(x, 2i),  sₖ ≜ ∑ᵢ₌₀ᵏ (1 if bᵢ = 0 else −1)

        k* ≜ min { k ∈ ℕ : sₖ < 0 }
        `flipExtra_0b1_Here` = 2^(2k*)
        ```

    References
    ----------
    [1] Jensen, I. (2000). A transfer matrix approach to the enumeration of plane meanders.
        Journal of Physics A: Mathematical and General, 33(34), 5953-5963.
        https://dx.doi.org/10.1088/0305-4470/33/34/301
    """
    findTheExtra_0b1: int = 0
    flipExtra_0b1_Here: int = 1
    while 0 <= findTheExtra_0b1:
        flipExtra_0b1_Here <<= 2
        if intWithExtra_0b1 & flipExtra_0b1_Here == 0:
            findTheExtra_0b1 += 1
        else:
            findTheExtra_0b1 += neg(1)
    return flipExtra_0b1_Here

def count(state: StateMeanders) -> StateMeanders:
    """Advance one meander transfer-matrix computation until `state.boundary` reaches zero.

    (AI generated docstring)

    You can use `count` to apply all transition rules for each boundary layer in `state` and update
    `state.lookupMeanders` in place. The transition set includes new-arc insertion, left-right
    crossing moves, and Dyck-path-based arc joining through `walkDyckPath`.

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    state : StateMeanders
        The same `state` instance after all boundary layers have been processed.

    Transfer Steps
    --------------
    The function decrements `state.boundary`, consumes each previous `arcCode`, and accumulates the
    next boundary dictionary. The transition for joining arches conditionally flips one Dyck-matched
    endpoint bit before packing the next `arcCode`.

    """
    while 0 < state.boundary:

        def analyzeArcCode(arcCode: int, meanders: int, state: StateMeanders=state) -> None:
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
            if bitsAlfaHasArcs and bitsZuluHasArcs and (bitsAlfaIsEven or bitsZuluIsEven):
                if bitsAlfaIsEven and (not bitsZuluIsEven):
                    bitsAlfa ^= walkDyckPath(bitsAlfa)
                elif bitsZuluIsEven and (not bitsAlfaIsEven):
                    bitsZulu ^= walkDyckPath(bitsZulu)
                arcCodeAnalysis = (bitsZulu >> 2 << 3 | bitsAlfa) >> 2
                if arcCodeAnalysis < state.arcCodeMaximum:
                    arcCodeAnalysis = min(arcCodeAnalysis, reflectArcCode(arcCodeAnalysis))
                    state.lookupMeanders[arcCodeAnalysis] = state.lookupMeanders.get(arcCodeAnalysis, 0) + meanders
        state.reduceBoundary()
        lookupArcCodeMeanders: dict[int, int] = state.lookupMeanders.copy()
        state.lookupMeanders = {}
        tuple(map(analyzeArcCode, lookupArcCodeMeanders.keys(), lookupArcCodeMeanders.values(), strict=True))
    return state
if TYPE_CHECKING:
    from collections.abc import Iterator
    from mapFolding.dataBaskets import StateMeanders

def getBitsLocator(arcCode: int) -> int:
    return (1 << (arcCode.bit_length() | 1) + 1) // 3

def reflectArcCode(arcCode: int) -> int:
    bitsLocator: int = getBitsLocator(arcCode)
    return (arcCode & bitsLocator) << 1 | arcCode >> 1 & bitsLocator

def pairTerminals(bitsSide: int, boxOfTerminals: Iterator[int], lookupPartner: list[int]) -> list[int]:
    boxOfOpen: list[int] = []
    boxOfCross: list[int] = []
    while 1 < bitsSide:
        terminal: int = next(boxOfTerminals)
        if bitsSide & 1 == 0:
            boxOfOpen.append(terminal)
        elif boxOfOpen:
            lookupPartner[terminal] = boxOfOpen.pop()
            lookupPartner[lookupPartner[terminal]] = terminal
        else:
            boxOfCross.append(terminal)
        bitsSide >>= 2
    return boxOfCross

def makeLookupPartner(arcCode: int) -> list[int]:
    bitsLocator: int = getBitsLocator(arcCode)
    bitsAlfa: int = arcCode & bitsLocator
    bitsZulu: int = arcCode >> 1 & bitsLocator
    lookupPartner: list[int] = list(range((bitsAlfa.bit_length() + bitsZulu.bit_length()) // 2 - 1))
    boxOfCrossAlfa: list[int] = pairTerminals(bitsAlfa, iter(range(len(lookupPartner))), lookupPartner)
    boxOfCrossZulu: list[int] = pairTerminals(bitsZulu, iter(range(len(lookupPartner) - 1, -1, -1)), lookupPartner)
    tuple(map(lookupPartner.__setitem__, boxOfCrossAlfa, boxOfCrossZulu, strict=True))
    tuple(map(lookupPartner.__setitem__, boxOfCrossZulu, boxOfCrossAlfa, strict=True))
    return lookupPartner

def isAxisMirrored吗(arcCode: int) -> bool:
    bitsLocator: int = getBitsLocator(arcCode)
    return (arcCode >> 1 & bitsLocator).bit_length() <= (arcCode & bitsLocator).bit_length() <= (arcCode >> 1 & bitsLocator).bit_length() + 4

def isHalfJoined吗(arcCode: int) -> bool:
    return isAxisMirrored吗(arcCode) and isJoined吗(makeLookupPartner(arcCode))

def doTheNeedful(state: StateMeanders) -> StateMeanders:
    state = count(state)
    state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
    state.lookupMeanders = {}
    return state
