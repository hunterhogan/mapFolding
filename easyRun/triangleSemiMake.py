#=SIN= Type-checker suppression: the IDE-selected flow remains a literal until the user edits it.
# pyright: reportUnnecessaryComparison=false
from __future__ import annotations

from hunterMakesPy import oneIndexed
from itertools import repeat
from mapFolding.algorithms.matrixMeanders import doTheNeedful
from mapFolding.algorithms.matrixMeandersTriangle import countDiagonal
from mapFolding.dataBaskets import StateMeanders
from mapFolding.kitFilesystem import appendStringToHere, writeDiagonal
from mapFolding.oeis import formatBFile
from mapFolding.oeis._byFormulaLookup import _A005315, _A005316
from research.matrixMeanders.formulasTriangle import boxOfDiagonals
from research.matrixMeanders.infoBooth import makePathFilenameDiagonal, pathFilenameTriangleSemiText
from tqdm.auto import tqdm

def writeDiagonalCount(n: int, 次diagonal: int) -> None:
	writeDiagonal({n: countDiagonal(n, 次diagonal)}, makePathFilenameDiagonal(次diagonal), 次diagonal, append=True)

def writeTriangleRows(nT: int, nStart: int, nStop: int, kStart: int = 0) -> None:
	pathFilenameTriangleSemiText.parent.mkdir(parents=True, exist_ok=True)
	#=SIN= For loop: the requested row writer preserves ordered output across its selected rows.
	for n in tqdm(range(nStart, nStop), position=0, leave=True, disable=True):
		#=SIN= For loop: each selected cell must be written in its triangle order.
		for k in tqdm(range(kStart, n // 2), position=1, leave=False, disable=False):
			nT += 1
			aOFn下k: int = 0
			if k == 0:
				aOFn下k = _A005315((n + 1) // 2)
			elif k in range(n // 2 - len(boxOfDiagonals), n // 2):
				diagonal: int = n // 2 - k
				aOFn下k = boxOfDiagonals[diagonal - oneIndexed](n)
			elif k == 1 and n % 2 == 0:
				aOFn下k = _A005315(n // 2 + 1) - 2 * _A005316(n)
			else:
				arcCode: int = 4 + (12 * (n & 1))
				arcCode = (arcCode << (4 * k)) - 1
				lookupMeanders: dict[int, int] = {arcCode: 1}
				state: StateMeanders = StateMeanders(n, 'semi', lookupMeanders=lookupMeanders)
				aOFn下k = doTheNeedful(state)
			appendStringToHere(formatBFile({nT: aOFn下k}), pathFilenameTriangleSemiText)

if __name__ == '__main__':
	flow = 'rows'
	flow: str = 'diagonal'
	if flow == 'rows':
		nT: int = 491
		nStart: int = 45
		nStop: int = 46
		kStart: int = 6
		writeTriangleRows(nT, nStart, nStop, kStart)
	else:
		次diagonal: int = 11
		nStart = 289
		nStop = 406
		tuple(map(writeDiagonalCount, tqdm(range(nStart, nStop)), repeat(次diagonal)))
"""
source /home/hunte/mapFolding/.venv/bin/activate && cd mapFolding
sudo nice -n -10 /home/hunte/mapFolding/.venv/bin/python -X faulthandler=0 -X tracemalloc=0 -X frozen_modules=on /home/hunte/mapFolding/easyRun/triangleSemiMake.py
"""
