from __future__ import annotations

from functools import partial
from humpy_cytoolz import juxt
from itertools import chain, repeat
from mapFolding.kitFilesystem import filterPathAbsent, makePathFilename
from oeis_tools import oeis_bfile
from pathlib import PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding.oeis._theTypes import OEISid
	from pathlib import Path

def makePathFilename_bFile[形PurePath: PurePath](oeisID: OEISid, path_oeisdata: 形PurePath) -> 形PurePath:
	return makePathFilename(path_oeisdata / 'files' / oeisID[0:4], oeis_bfile(oeisID))

def makePathFilenameEntry[形PurePath: PurePath](oeisID: OEISid, path_oeisdata: 形PurePath) -> 形PurePath:
	return makePathFilename(path_oeisdata / 'seq' / oeisID[0:4], oeisID, suffix='.seq')

def makeBoxOfPathFilename[形PurePath: PurePath](boxOfOEISid: Iterable[OEISid], path_oeisdata: 形PurePath) -> Iterator[形PurePath]:
	return chain.from_iterable(map(juxt(makePathFilenameEntry, makePathFilename_bFile), boxOfOEISid, repeat(path_oeisdata), strict=False))

def filterEntryAbsent(boxOfOEISid: Iterable[OEISid], path_oeisdata: Path) -> tuple[OEISid, ...]:
	return filterPathAbsent(boxOfOEISid, partial(makePathFilenameEntry, path_oeisdata=path_oeisdata))
