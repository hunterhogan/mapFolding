"""
Identifiers
    This module has two sets of identifiers. One set is active, and the other set is in uniformly formatted comments
    at the end of every line that includes an identifier that has an alternative identifier. 

    First, that might be distracting. In Visual Studio Code, the following extension will hide all comments but not docstrings:
    https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-hide-comments

    Second, you can swap the sets of identifiers or delete one set of identifiers permanently.

    Step 1: regex find:
"""
# ^(?! *#)( *?)(\S.+?)( # )(.+) # This line is a comment and not a docstring because the Python interpreter handles `\S` better in a comment
"""
    Step 2: choose a regex replace option:
        A) To SWAP the sets of identifiers
        $1$4$3$2
        B) To PERMANENTLY replace the active set of identifiers
        $1$4
        C) To PERMANENTLY delete the inactive set of identifiers, which are in the comments
        $1$2
"""
from typing import List

from mapFolding import outfitFoldings, validateTaskDivisions

def foldings(p: List[int], mod: int = 0, res: int = 0) -> int: # def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    p, n, D, s, gap = outfitFoldings(p) # listDimensions, leavesTotal, connectionGraph, track, potentialGaps = outfitFoldings(listDimensions)
    mod, res = validateTaskDivisions(mod, res, n) # computationDivisions, computationIndex = validateTaskDivisions(computationDivisions, computationIndex, leavesTotal)

    d: int = len(p) # dimensionsTotal: int = len(listDimensions)

    from mapFolding.lovelace import countFoldings
    foldingsTotal = countFoldings(
        s, gap, D, # track, potentialGaps, connectionGraph,
        n, d, # leavesTotal, dimensionsTotal,
        mod, res # computationDivisions, computationIndex
        )

    return foldingsTotal
