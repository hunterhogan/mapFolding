from __future__ import annotations

from functools import partial
from humpy_cytoolz import compose, juxt
from itertools import chain, filterfalse, repeat
from mapFolding.kitFilesystem import readText, writeText
from more_itertools import consume
from oeis_tools import oeis_bfile
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Iterable, Iterator
	from mapFolding.theTypes import OEISid
	from pathlib import PurePosixPath

def makePathFilename_bFile[形PurePath: PurePath](oeisID: OEISid, path_oeisdata: 形PurePath) -> 形PurePath:
	return path_oeisdata / 'files' / oeisID[0:4] / oeis_bfile(oeisID)

def makePathFilenameEntry[形PurePath: PurePath](oeisID: OEISid, path_oeisdata: 形PurePath) -> 形PurePath:
	return path_oeisdata / 'seq' / oeisID[0:4] / f"{oeisID}.seq"

def makeBoxOfPathFilename[形PurePath: PurePath](boxOfOEISid: Iterable[OEISid], path_oeisdata: 形PurePath) -> Iterator[形PurePath]:
	return chain.from_iterable(map(juxt(makePathFilenameEntry, makePathFilename_bFile), boxOfOEISid, repeat(path_oeisdata), strict=False))

def filterEntryAbsent(boxOfOEISid: Iterable[OEISid], path_oeisdata: Path) -> tuple[OEISid, ...]:
	return (*filterfalse(compose(Path.exists, partial(makePathFilenameEntry, path_oeisdata=path_oeisdata)), boxOfOEISid),)

def directoryEmpty吗(pathDirectory: Path) -> bool:
	return pathDirectory.is_dir() and next(pathDirectory.iterdir(), None) is None

def removeDirectoriesEmpty(pathTerms: Path) -> Path:
	"""Remove each empty directory, such as `files/A400`, that no b-file in `pathTerms` occupies any longer.

	(AI generated docstring)

	`Path.rmdir` removes only an empty directory, so a directory that holds any file stays.
	"""
	consume(map(Path.rmdir, filter(directoryEmpty吗, chain.from_iterable(map(Path.iterdir, filter(Path.is_dir, pathTerms.iterdir()))))))
	return pathTerms

def copy_oeisdata(boxOfRelativePath: Iterable[PurePosixPath], path_oeisdata: Path, pathSnapshot: Path) -> Path:
	boxOfRelativePath = tuple(filter(compose(Path.exists, path_oeisdata.joinpath), boxOfRelativePath))
	consume(map(writeText, map(compose(readText, path_oeisdata.joinpath), boxOfRelativePath), map(pathSnapshot.joinpath, boxOfRelativePath), strict=True))
	return pathSnapshot
