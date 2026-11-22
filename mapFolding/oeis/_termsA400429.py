from __future__ import annotations

from hunterMakesPy import oneIndexed
from mapFolding.algorithms.matrixMeanders import doTheNeedful
from mapFolding.dataBaskets import StateMeanders
from mapFolding.dataStructures import locateIndex, locateRow
from mapFolding.oeis._metadata import getMetadataTriangle, getValuesKnown, updateValuesKnown
from mapFolding.oeis.A400429 import boxOfDiagonals
from more_itertools import consume
from tqdm.auto import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from mapFolding.oeis._dataBaskets import MetadataOEISidTriangle
	from pathlib import Path

def countMeandersWinding(n: int, k: int) -> int:
	aOFn下k: int = 0
	if k == 0:
		aOFn下k = getValuesKnown('A005315')[(n + 1) // 2]
	elif k in range(n // 2 - len(boxOfDiagonals), n // 2):
		diagonal: int = n // 2 - k
		aOFn下k = boxOfDiagonals[diagonal - oneIndexed](n)
	elif k == 1 and n % 2 == 0:
		aOFn下k = getValuesKnown('A005315')[n // 2 + 1] - 2 * getValuesKnown('A005316')[n]
	else:
		arcCode: int = 4 + (12 * (n & 1))
		arcCode = (arcCode << (4 * k)) - 1
		lookupMeanders: dict[int, int] = {arcCode: 1}
		state: StateMeanders = StateMeanders(n, 'semi', lookupMeanders=lookupMeanders)
		state = doTheNeedful(state)
		aOFn下k = state.totalMeanders
	return aOFn下k

def writeTerm(index: int) -> Path:
	metadata: MetadataOEISidTriangle = getMetadataTriangle('A400429')
	n, k = locateIndex(index, metadata['rowLength'], metadata['rowStart'], metadata['offset'])
	return updateValuesKnown('A400429', {index: countMeandersWinding(n, k)})

def writeTriangleRows(nStop: int) -> None:
	metadata: MetadataOEISidTriangle = getMetadataTriangle('A400429')
	consume(map(writeTerm, tqdm(range(
		metadata['valueUnknown']
		, locateRow(nStop, metadata['rowLength'], metadata['rowStart'], metadata['offset']).start
	))))
