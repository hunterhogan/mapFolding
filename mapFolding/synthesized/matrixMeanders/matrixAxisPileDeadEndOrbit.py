"""Count by the pile cut with the dead-end start, with one key for each reflection orbit.

(AI generated docstring)

The module adapts `mapFolding.algorithms.matrixAxisPileDeadEnd`.
`mapFolding.kitAST.matrixMeanders.makeModulesAxis.makeOrbit` generates this module.
Regenerate the module instead of editing it.

Contents
--------
Functions
    count
        Advance the pile-cut transfer with the dead-end start to the symmetry axis.
    doTheNeedful
        Count the halves that join at the symmetry axis and add the count to `state.countAddend`.
    endStrandAtDeadEnd
        End the innermost strand of `bitsAlfa` at the dead end, and mark the start of the strand.
    isHalfJoined吗
        Decide whether the half that a placed state describes joins with its half-turn image.
    isJoined吗
        Decide whether `lookupPartner` and its half-turn image form one strand from the start.
    makeLookupPartner
        Pair each terminal of `word` with the other end of its arc, in circle order.
    reflectArcCode
        Reflect a state before or after the placement across the road.
"""
from __future__ import annotations

from mapFolding.algorithms.matrixAxisPile import getBitsLocator, pairTerminals
from mapFolding.algorithms.matrixMeanders import walkDyckPath
from operator import add, mul
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from mapFolding.dataBaskets import StateMeandersAxis

def reflectArcCode(arcCode: int) -> int:
    """Reflect a state before or after the placement across the road.

    (AI generated docstring)

    The reflection σ exchanges `bitsAlfa` and `bitsZulu` of the `arcCode`, or of `word` and of
    `bitStart` in a placed state.

    Parameters
    ----------
    arcCode : int
        State of the transfer. A negative value is a placed state.

    Returns
    -------
    arcCodeReflected : int
        State of the reflected half, of the same kind as `arcCode`.
    """
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
    """End the innermost strand of `bitsAlfa` at the dead end, and mark the start of the strand.

    (AI generated docstring)

    The innermost terminal of `bitsAlfa` and its partner leave the word, and the place of the partner
    becomes the start of the strand. For the innermost terminal of `bitsZulu`, exchange the arguments
    and reflect the result with `reflectArcCode`.

    Parameters
    ----------
    bitsAlfa : int
        Side of the road whose innermost terminal ends at the dead end.
    bitsZulu : int
        Other side of the road.

    Returns
    -------
    arcCodePlaced : int
        Placed state, `~(word << word.bit_length() | bitStart)`.
    """
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
    """Advance the pile-cut transfer with the dead-end start to the symmetry axis.

    (AI generated docstring)

    You can use this function to obtain, for each state at the symmetry axis, the number of halves
    that end with that state. A state before the placement takes the moves of
    `mapFolding.algorithms.matrixMeanders.count` and the moves that place the dead end. A placed state
    takes the same moves on `word`, and `bitStart` shifts with its side. Kind `'pileDeadEnd'` places
    the dead end beside any state; kind `'pileDeadEndOutward'` places it only where its side has no
    terminals.

    Parameters
    ----------
    state : StateMeandersAxis
        Transfer state of kind `'pileDeadEnd'` or `'pileDeadEndOutward'`. `state.lookupMeanders` maps
        each state to its count.

    Returns
    -------
    state : StateMeandersAxis
        The same state. `state.lookupMeanders` holds the counts at the symmetry axis.
    """
    while 0 < state.boundary:

        def analyzeArcCode(arcCode: int, meanders: int, state: StateMeandersAxis = state) -> None:
            """Accumulate one pile-cut state's successors with an optional dead-end start.

            (AI generated docstring)

            Input Mutation
            --------------
            `state.lookupMeanders` receives `meanders` for each allowed successor below
            `state.arcCodeMaximum`. Existing successor counts increase by that amount.

            Parameters
            ----------
            arcCode : int
                Packed pile-cut state. Negative values record a previously placed dead end.
            meanders : int
                Number of halves represented by `arcCode`.
            state : StateMeandersAxis
                Enclosing transfer state by default, with the boundary already reduced.
                Its kind determines where an unplaced dead end may be placed.
            """
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
    """Pair each terminal of `word` with the other end of its arc, in circle order.

    (AI generated docstring)

    The circle order is the order of `mapFolding.algorithms.matrixAxisPile.makeLookupPartner`, and the
    terminals of `word` skip the circle position of the start.

    Parameters
    ----------
    arcCode : int
        `word` of a placed state.
    terminal : int
        Circle position of the start of the strand. The start has no partner in `word`.

    Returns
    -------
    lookupPartner : list[int]
        List of N circle positions. Entry i is the partner of circle position i, and the entry of the
        start is the start.
    """
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
    """Decide whether `lookupPartner` and its half-turn image form one strand from the start.

    (AI generated docstring)

    The symmetry axis identifies circle position i with circle position N − i. The walk starts at the
    start of the strand and alternates between the identification at the axis and an arc of the half
    until the walk reaches the center terminal, position 0. The half joins when the walk uses every
    strand of the half.

    Parameters
    ----------
    lookupPartner : Sequence[int]
        Partner of each of the N circle positions.
    terminal : int
        Circle position of the start of the strand.

    Returns
    -------
    isJoined : bool
        `True` when the half and its half-turn image form one strand.
    """
    strands: int = 1
    while terminal != 0:
        terminal = lookupPartner[len(lookupPartner) - terminal]
        strands += 1
    return 2 * strands - 1 == len(lookupPartner)

def isHalfJoined吗(arcCode: int) -> bool:
    """Decide whether the half that a placed state describes joins with its half-turn image.

    (AI generated docstring)

    The test takes the center terminal on the side of `bitsAlfa`. The half joins when it is a placed
    state, the side of `bitsAlfa` has one more terminal than the side of `bitsZulu`, and `isJoined吗`
    accepts the walk from the start.

    Parameters
    ----------
    arcCode : int
        State of the transfer at the symmetry axis. A state before the placement never joins.

    Returns
    -------
    isHalfJoined : bool
        `True` when the half and its half-turn image form one symmetric object.
    """
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
    """Count the halves that join at the symmetry axis and add the count to `state.countAddend`.

    (AI generated docstring)

    You can use this function to complete a term of A007822 with kind `'pileDeadEnd'` or a term of
    A223096 with kind `'pileDeadEndOutward'`. With D(w) the count of state w at the symmetry axis, J
    the test `isHalfJoined吗`, and σ the reflection `reflectArcCode`, the term is
    Σ_w D(w)·(J(w) + J(σw)) / 2. The sum is exact because the transfer places the dead end on both
    sides of the road.

    Parameters
    ----------
    state : StateMeandersAxis
        Transfer state of kind `'pileDeadEnd'` or `'pileDeadEndOutward'` before the transfer.

    Returns
    -------
    state : StateMeandersAxis
        The same state with the term in `state.countAddend` and an empty `state.lookupMeanders`. Read
        the term from `state.totalMeanders`.
    """
    state = count(state)
    state.countAddend += sum(map(mul, state.lookupMeanders.values(), map(add, map(isHalfJoined吗, state.lookupMeanders.keys()), map(isHalfJoined吗, map(reflectArcCode, state.lookupMeanders.keys())), strict=True), strict=True)) // 2
    state.lookupMeanders = {}
    return state
