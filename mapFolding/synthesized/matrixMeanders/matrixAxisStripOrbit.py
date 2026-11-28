"""Count symmetric foldings of a stamp strip by the strip cut, with one key per reflection orbit.

(AI generated docstring)

The module adapts `mapFolding.algorithms.matrixAxisStrip`.
`mapFolding.kitAST.matrixMeanders.makeModulesAxis.makeOrbit` generates this module.
Regenerate the module instead of editing it.

Contents
--------
Functions
    count
        Advance the strip-cut transfer from the first crossing to the symmetry axis.
    doTheNeedful
        Count the halves that join at the symmetry axis and add the count to `state.countAddend`.
    isHalfJoined吗
        Decide whether the half that a circle word describes joins with its half-turn image.
    isJoined吗
        Decide whether the arcs in `lookupPartner` and their half-turn image form one strand.
    makeLookupPartner
        Pair each terminal of a circle word with the terminal at the other end of its arc.
    reflectArcCode
        Reflect a circle word across the strip.
    walkDyckPathAlfa
        Locate the partner of position 0 in a circle word.
    walkDyckPathZulu
        Locate the partner of position N − 1 in a circle word.
"""
from __future__ import annotations

from operator import add, mul, neg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from mapFolding.dataBaskets import StateMeandersAxis

def walkDyckPathAlfa(arcCode: int) -> int:
    """Locate the partner of position 0 in a circle word.

    (AI generated docstring)

    Parameters
    ----------
    arcCode : int
        Circle word of the strip cut, with the sentinel bit at position N.

    Returns
    -------
    bitPartner : int
        Integer with one set bit, at the position of the terminal that shares an arc with position 0.
    """
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
    """Locate the partner of position N − 1 in a circle word.

    (AI generated docstring)

    Parameters
    ----------
    arcCode : int
        Circle word of the strip cut, with the sentinel bit at position N.

    Returns
    -------
    bitPartner : int
        Integer with one set bit, at the position of the terminal that shares an arc with position N −
        1.
    """
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
    """Reflect a circle word across the strip.

    (AI generated docstring)

    The reflection σ reverses the order of the terminals and exchanges opening and closing terminals,
    so the terminal of bank 1 beside the gate becomes position 0.

    Parameters
    ----------
    arcCode : int
        Circle word of the strip cut, with the sentinel bit at position N.

    Returns
    -------
    arcCodeReflected : int
        Circle word of the reflected half, with the sentinel bit at position N.
    """
    return int(format(arcCode, 'b')[:0:-1], 2) ^ (1 << arcCode.bit_length()) - 1

def count(state: StateMeandersAxis) -> StateMeandersAxis:
    """Advance the strip-cut transfer from the first crossing to the symmetry axis.

    (AI generated docstring)

    You can use this function to obtain, for each circle word at the symmetry axis, the number of
    halves that end with that circle word. Each step crosses one more pile of the strip. Each step
    applies four moves to each circle word: both new terminals go out, the terminal of bank 0 comes
    in, the terminal of bank 1 comes in, or both terminals come in. Each step reduces `state.boundary`
    by one, and the transfer stops when `state.boundary` is 0.

    Parameters
    ----------
    state : StateMeandersAxis
        Transfer state of kind `'strip'`. `state.lookupMeanders` maps each circle word to its count.

    Returns
    -------
    state : StateMeandersAxis
        The same state. `state.lookupMeanders` holds the counts at the symmetry axis.
    """
    while 0 < state.boundary:

        def analyzeArcCode(arcCode: int, meanders: int, state: StateMeandersAxis = state) -> None:
            """Accumulate one circle word's successors across the next pile of the strip.

            (AI generated docstring)

            Input Mutation
            --------------
            `state.lookupMeanders` receives `meanders` for each allowed successor below
            `state.arcCodeMaximum`. Existing successor counts increase by that amount.

            Parameters
            ----------
            arcCode : int
                Circle word with a sentinel bit above the terminal bits.
            meanders : int
                Number of halves represented by `arcCode`.
            state : StateMeandersAxis
                Enclosing strip-cut state by default, with the boundary already reduced.
            """
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
    """Pair each terminal of a circle word with the terminal at the other end of its arc.

    (AI generated docstring)

    Parameters
    ----------
    arcCode : int
        Circle word of the strip cut, with the sentinel bit at position N.

    Returns
    -------
    lookupPartner : list[int]
        List of N positions. `lookupPartner[terminal]` is the position of the terminal that shares an
        arc with `terminal`.
    """
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
    """Decide whether the arcs in `lookupPartner` and their half-turn image form one strand.

    (AI generated docstring)

    The symmetry axis identifies position i with position N − i. The walk starts at the center
    terminal, position 0, and alternates between an arc of the half and the identification at the axis
    until the walk reaches position N / 2. The half joins when the walk uses every one of the N / 2
    strands of the half.

    Parameters
    ----------
    lookupPartner : Sequence[int]
        Partner of each of the N positions on the cut, in circle order.

    Returns
    -------
    isJoined : bool
        `True` when the half and its half-turn image form one strand.
    """
    terminal: int = lookupPartner[0]
    strands: int = 1
    while terminal != len(lookupPartner) // 2:
        terminal = lookupPartner[len(lookupPartner) - terminal]
        strands += 1
    return 2 * strands == len(lookupPartner)

def isHalfJoined吗(arcCode: int) -> bool:
    """Decide whether the half that a circle word describes joins with its half-turn image.

    (AI generated docstring)

    Parameters
    ----------
    arcCode : int
        Circle word of the strip cut at the symmetry axis.

    Returns
    -------
    isHalfJoined : bool
        `True` when the half and its half-turn image form one strand.
    """
    return isJoined吗(makeLookupPartner(arcCode))

def doTheNeedful(state: StateMeandersAxis) -> StateMeandersAxis:
    """Count the halves that join at the symmetry axis and add the count to `state.countAddend`.

    (AI generated docstring)

    You can use this function to complete a term of A007822. With D(w) the count of circle word w at
    the symmetry axis, J the test `isHalfJoined吗`, and σ the reflection `reflectArcCode`, the term is
    Σ_w D(w)·(J(w) + J(σw)) / 2. The symmetrized sum is exact for this module and for the variant
    whose keys are σ-orbit representatives.

    Parameters
    ----------
    state : StateMeandersAxis
        Transfer state of kind `'strip'` before the transfer.

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
