"""Transfer matrix algorithm implementations in NumPy (*Num*erical *Py*thon) and pandas.

Citations
---------
- https://github.com/hunterhogan/mapFolding/blob/main/citations/Jensen.bib
- https://github.com/hunterhogan/mapFolding/blob/main/citations/Howroyd.bib

See Also
--------
`matrixMeanders`: transfer matrix algorithm implementation in pure Python with `int` (*int*eger) contained in a `dict` (*dict*ionary).
https://oeis.org/A000682
https://oeis.org/A005316
https://github.com/archmageirvine/joeis/blob/5dc2148344bff42182e2128a6c99df78044558c5/src/irvine/oeis/a005/A005316.java
"""
from __future__ import annotations

from gc import collect as goByeBye
from mapFolding.algorithms.matrixMeandersShare import flipTheExtra_0b1, getTotalBuckets, integersWide吗, prune
from mapFolding.synthesized.matrixMeanders.pruneBigInt import countBigInt
from mapFolding.theTypes import 形ArcCode, 形Meanders
from research.matrixMeanders.formulasTriangle import boxOfDiagonals
from typing import TYPE_CHECKING
from warnings import warn
import pandas

if TYPE_CHECKING:
    from collections.abc import Callable
    from mapFolding.dataBaskets import StateMeanders

def pruneDataFrame(state: StateMeanders, dataframeAnalyzed: pandas.DataFrame) -> tuple[StateMeanders, pandas.DataFrame]:
    boundary: int = state.boundary + 1

    def removeKnownValue(次diagonal: int, 工: Callable[[int], int]) -> int:
        nonlocal dataframeAnalyzed
        arcCode: int = (1 << 2 * (boundary - 2 * 次diagonal) + 2) - 1
        selectorKnown: pandas.Series[bool] = dataframeAnalyzed['analyzed'].eq(arcCode)
        subtotalMeanders: int = sum(map(int, dataframeAnalyzed.loc[selectorKnown, 'meanders']))
        dataframeAnalyzed = dataframeAnalyzed.loc[~selectorKnown]
        return 工(boundary) * subtotalMeanders
    state.countAddend += sum(map(removeKnownValue, range(1, boundary // 2 + 1), boxOfDiagonals))
    if not len(dataframeAnalyzed.index):
        state.boundary = 0
    return (state, dataframeAnalyzed.reset_index(drop=True))

def count(state: StateMeanders) -> StateMeanders:
    """Count meanders with matrix transfer algorithm using pandas DataFrame.

    Parameters
    ----------
    state : StateMeanders
        The algorithm state containing current `boundary`, `lookupMeanders`, and thresholds.

    Returns
    -------
    state : StateMeanders
        Updated state with new `boundary` and `lookupMeanders`.
    """
    dataframeAnalyzed = pandas.DataFrame({'analyzed': pandas.Series(name='analyzed', data=state.lookupMeanders.keys(), copy=False, dtype=形ArcCode), 'meanders': pandas.Series(name='meanders', data=state.lookupMeanders.values(), copy=False, dtype=形Meanders)})
    state.lookupMeanders.clear()
    while 0 < state.boundary and (not integersWide吗(state, dataframe=dataframeAnalyzed)):

        def aggregateArcCodes() -> None:
            nonlocal dataframeAnalyzed, state
            dataframeAnalyzed = dataframeAnalyzed.iloc[0:state.次Target].groupby('analyzed', sort=False)['meanders'].aggregate('sum').reset_index()
            state, dataframeAnalyzed = pruneDataFrame(state, dataframeAnalyzed)

        def addLoop(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
            """Compute arcCode with the 'simple' formula.

            Formula
            -------
            ```python
                arcCode = ((bitsAlfa | (bitsZulu << 1)) << 2) | 3
            ```

            Notes
            -----
            Using `+= 3` instead of `|= 3` is valid in this specific case. Left shift by two means the
            last bits are '0b00'. '0 + 3' is '0b11', and '0b00 | 0b11' is also '0b11'.
            """
            dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']
            dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator
            bitsZulu: pandas.Series = dataframeMeanders['arcCode'].copy()
            bitsZulu //= 2 ** 1
            bitsZulu &= state.bitsLocator
            bitsZulu *= 2 ** 1
            dataframeMeanders.loc[:, 'analyzed'] |= bitsZulu
            del bitsZulu
            dataframeMeanders.loc[:, 'analyzed'] *= 2 ** 2
            dataframeMeanders.loc[:, 'analyzed'] += 3
            dataframeMeanders.loc[state.arcCodeMAXIMUM <= dataframeMeanders['analyzed'], 'analyzed'] = 0
            return dataframeMeanders

        def connectAcrossLine(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
            """Compute `arcCode` from `bitsAlfa` and `bitsZulu` if at least one is an even number.

            Before computing `arcCode`, some values of `bitsAlfa` and `bitsZulu` are modified.

            Warning
            -------
            This function deletes rows from `dataframeMeanders`. Always run this analysis last.

            Formula
            -------
            ```python
                if 1 < bitsAlfa and 1 < bitsZulu and (bitsAlfaIsEven or bitsZuluIsEven):
                    arcCode = (bitsAlfa >> 2) | ((bitsZulu >> 2) << 1)
            ```
            """
            dataframeMeanders['analyzed'] = dataframeMeanders['arcCode'].copy()
            dataframeMeanders['analyzed'] &= state.bitsLocator
            dataframeMeanders['analyzed'] = dataframeMeanders['analyzed'].gt(1)
            bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
            bitsTarget //= 2 ** 1
            bitsTarget &= state.bitsLocator
            dataframeMeanders['analyzed'] *= bitsTarget
            del bitsTarget
            dataframeMeanders = dataframeMeanders.loc[1 < dataframeMeanders['analyzed']]
            dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
            dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator
            dataframeMeanders.loc[:, 'analyzed'] &= 1
            bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
            bitsTarget //= 2 ** 1
            bitsTarget &= state.bitsLocator
            dataframeMeanders.loc[:, 'analyzed'] &= bitsTarget
            del bitsTarget
            dataframeMeanders.loc[:, 'analyzed'] ^= 1
            dataframeMeanders = dataframeMeanders.loc[0 < dataframeMeanders['analyzed']]
            dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
            dataframeMeanders.loc[:, 'analyzed'] //= 2 ** 1
            dataframeMeanders.loc[:, 'analyzed'] &= 1
            bitsTarget = dataframeMeanders['arcCode'].copy()
            bitsTarget &= state.bitsLocator
            bitsTarget.loc[0 < dataframeMeanders['analyzed']] = flipTheExtra_0b1(bitsTarget.loc[0 < dataframeMeanders['analyzed']]).astype(形ArcCode)
            dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode'].copy()
            dataframeMeanders.loc[:, 'analyzed'] //= 2 ** 1
            dataframeMeanders.loc[:, 'analyzed'] &= state.bitsLocator
            dataframeMeanders.loc[0 < dataframeMeanders.loc[:, 'arcCode'] & 1, 'analyzed'] = flipTheExtra_0b1(dataframeMeanders.loc[0 < dataframeMeanders.loc[:, 'arcCode'] & 1, 'analyzed']).astype(形ArcCode)
            dataframeMeanders.loc[:, 'analyzed'] //= 2 ** 2
            dataframeMeanders.loc[:, 'analyzed'] *= 2 ** 3
            dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget
            del bitsTarget
            dataframeMeanders.loc[:, 'analyzed'] //= 2 ** 2
            dataframeMeanders.loc[state.arcCodeMAXIMUM <= dataframeMeanders['analyzed'], 'analyzed'] = 0
            return dataframeMeanders

        def dragUp(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
            """Compute `arcCode` from `bitsAlfa`.

            Formula
            -------
            ```python
                if 1 < bitsAlfa:
                    arcCode = ((1 - (bitsAlfa & 1)) << 1) | (bitsZulu << 3) | (bitsAlfa >> 2)
                # `(1 - (bitsAlfa & 1)` is an evenness test.
            ```
            """
            dataframeMeanders['analyzed'] = dataframeMeanders['arcCode']
            dataframeMeanders.loc[:, 'analyzed'] &= 1
            dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']
            dataframeMeanders.loc[:, 'analyzed'] *= 2 ** 1
            bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
            bitsTarget //= 2 ** 1
            bitsTarget &= state.bitsLocator
            bitsTarget *= 2 ** 3
            dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget
            del bitsTarget
            'NOTE In this code block, I rearranged the "formula" to use `bitsTarget` for two goals.\n            1. `(bitsAlfa >> 2)`.\n            2. `if 1 < bitsAlfa`. The trick is in the equivalence of v1 and v2.\n\n            v1: BITScow | (BITSwalk >> 2)\n            v2: ((BITScow << 2) | BITSwalk) >> 2\n\n            The "formula" calls for v1, but by using v2, `bitsTarget` is not changed. Therefore, because `bitsTarget` is\n            `bitsAlfa`, I can use `bitsTarget` for goal 2, `if 1 < bitsAlfa`.\n            '
            dataframeMeanders.loc[:, 'analyzed'] *= 2 ** 2
            bitsTarget = dataframeMeanders['arcCode'].copy()
            bitsTarget &= state.bitsLocator
            dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget
            dataframeMeanders.loc[:, 'analyzed'] //= 2 ** 2
            dataframeMeanders.loc[bitsTarget <= 1, 'analyzed'] = 0
            del bitsTarget
            dataframeMeanders.loc[state.arcCodeMAXIMUM <= dataframeMeanders['analyzed'], 'analyzed'] = 0
            return dataframeMeanders

        def dragDown(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
            """Compute `arcCode` from `bitsZulu`.

            Formula
            -------
            ```python
                if 1 < bitsZulu:
                    arcCode = (1 - (bitsZulu & 1)) | (bitsAlfa << 2) | (bitsZulu >> 1)
            ```
            """
            dataframeMeanders.loc[:, 'analyzed'] = dataframeMeanders['arcCode']
            dataframeMeanders.loc[:, 'analyzed'] //= 2 ** 1
            dataframeMeanders.loc[:, 'analyzed'] &= 1
            dataframeMeanders.loc[:, 'analyzed'] &= 1
            dataframeMeanders.loc[:, 'analyzed'] = 1 - dataframeMeanders.loc[:, 'analyzed']
            bitsTarget: pandas.Series = dataframeMeanders['arcCode'].copy()
            bitsTarget &= state.bitsLocator
            bitsTarget *= 2 ** 2
            dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget
            del bitsTarget
            dataframeMeanders.loc[:, 'analyzed'] *= 2 ** 1
            bitsTarget = dataframeMeanders['arcCode'].copy()
            bitsTarget //= 2 ** 1
            bitsTarget &= state.bitsLocator
            dataframeMeanders.loc[:, 'analyzed'] |= bitsTarget
            dataframeMeanders.loc[:, 'analyzed'] //= 2 ** 1
            dataframeMeanders.loc[bitsTarget <= 1, 'analyzed'] = 0
            del bitsTarget
            dataframeMeanders.loc[state.arcCodeMAXIMUM <= dataframeMeanders['analyzed'], 'analyzed'] = 0
            return dataframeMeanders

        def recordArcCodes(dataframeMeanders: pandas.DataFrame) -> pandas.DataFrame:
            """Abstraction makes it easier to do things such as write to disk."""
            nonlocal dataframeAnalyzed
            次StopAnalyzed: int = state.次Target + int((0 < dataframeMeanders['analyzed']).sum())
            if state.次Target < 次StopAnalyzed:
                if len(dataframeAnalyzed.index) < 次StopAnalyzed:
                    warn(f'Lengthened `dataframeAnalyzed` from {len(dataframeAnalyzed.index)} to 次StopAnalyzed={次StopAnalyzed!r}; n={state.n}, state.boundary={state.boundary!r}.', stacklevel=2)
                    dataframeAnalyzed = dataframeAnalyzed.reindex(index=pandas.RangeIndex(次StopAnalyzed), fill_value=0)
                dataframeAnalyzed.loc[state.次Target:次StopAnalyzed - 1, ['analyzed']] = dataframeMeanders.loc[0 < dataframeMeanders['analyzed'], ['analyzed']].to_numpy(dtype=形ArcCode, copy=False)
                dataframeAnalyzed.loc[state.次Target:次StopAnalyzed - 1, ['meanders']] = dataframeMeanders.loc[0 < dataframeMeanders['analyzed'], ['meanders']].to_numpy(dtype=形Meanders, copy=False)
                state.次Target = 次StopAnalyzed
            del 次StopAnalyzed
            return dataframeMeanders
        dataframeMeanders: pandas.DataFrame = pandas.DataFrame({'arcCode': pandas.Series(name='arcCode', data=dataframeAnalyzed['analyzed'], copy=False, dtype=形ArcCode), 'analyzed': pandas.Series(name='analyzed', data=0, dtype=形ArcCode), 'meanders': pandas.Series(name='meanders', data=dataframeAnalyzed['meanders'], copy=False, dtype=形Meanders)})
        del dataframeAnalyzed
        goByeBye()
        state.bitWidth = int(dataframeMeanders['arcCode'].max()).bit_length()
        state.setBitsLocator()
        length: int = getTotalBuckets(state, len(dataframeMeanders.index))
        dataframeAnalyzed = pandas.DataFrame({'analyzed': pandas.Series(name='analyzed', data=0, index=pandas.RangeIndex(length), dtype=形ArcCode), 'meanders': pandas.Series(name='meanders', data=0, index=pandas.RangeIndex(length), dtype=形Meanders)}, index=pandas.RangeIndex(length))
        state.boundary -= 1
        state.set_arcCodeMAXIMUM()
        state.次Target = 0
        dataframeMeanders = addLoop(dataframeMeanders)
        dataframeMeanders = recordArcCodes(dataframeMeanders)
        dataframeMeanders = dragUp(dataframeMeanders)
        dataframeMeanders = recordArcCodes(dataframeMeanders)
        dataframeMeanders = dragDown(dataframeMeanders)
        dataframeMeanders = recordArcCodes(dataframeMeanders)
        dataframeMeanders = connectAcrossLine(dataframeMeanders)
        dataframeMeanders = recordArcCodes(dataframeMeanders)
        del dataframeMeanders
        goByeBye()
        aggregateArcCodes()
    state.lookupMeanders = dataframeAnalyzed.set_index('analyzed')['meanders'].to_dict()
    del dataframeAnalyzed
    return state

def doTheNeedful(state: StateMeanders) -> StateMeanders:
    """Compute `meanders` with a transfer matrix algorithm implemented in pandas.

    Parameters
    ----------
    state : StateMeanders
        The algorithm state.

    Returns
    -------
    state : StateMeanders
        The completed meander transfer state.
    """
    state = prune(state)
    while 0 < state.boundary:
        if integersWide吗(state):
            state = countBigInt(state)
        else:
            state = count(state)
    return state
