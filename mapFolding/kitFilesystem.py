#ruff: file-ignore[suspicious-pickle-usage] #=Sin= Centralize all pickle usage here.
"""Name, save, cache, and load folding computation data.

(AI generated docstring)

You can use this module to persist counts, albums, integer triangles, and diagonals. This module
also retrieves cached text from URLs.

Contents
--------
Functions
	Create paths and filenames
		getPathRootJobDEFAULT
			Create the default job directory.
		makeFilenameArrayFoldings
			Build a filename for sampled array-folding data.
		makeFilenameCount
			Join positional and named filename components.
		makeFilenameFolds
			Encode map dimensions in their supplied order.
		makePathFilenameArrayFoldings
			Combine a sample-data directory and filename.
		makePathFilenameCount
			Resolve an output path and create parent directories.
		makePathFilenameFolds
			Resolve an output path named from map dimensions.
	Confirm write capability
		saveTotalFAILearly
			Leave a test value at a new output path before counting.
	Write
		appendTriangle
			Append numbered dense groups to a triangle CSV file.
		iterableToCSV
			Write groups to CSV with optional appending.
		saveTotal
			Save an integer with fallback output on failure.
		saveTriangleGroup
			Save one dense group using a generated or supplied filename.
		writeAlbum
			Write each folding as one CSV record.
		writeDiagonal
			Write sorted group, diagonal, and value records.
		writeText
			Write text and create missing parent directories.
		writeTriangle
			Write sorted group keys followed by group values.
		writeTriangleSparse
			Write sparse groups with empty fields for missing cells.
	Read and write
		getCacheOrURL
			Retrieve cached text or request updated text from a URL.
	Read
		getDataFrameFoldings
			Load bundled folding data for an elimination state.
		readAlbum
			Load all folding records into a tuple.
		readDataFrame
			Load serialized data as a dataframe.
		readDiagonal
			Read one diagonal from group, diagonal, and value records.
		readText
			Read UTF-8 text.
		readTriangle
			Read numbered integer groups.
		readTriangleSparse
			Read sparse integer groups without shifting missing columns.
	Stream
		streamAlbum
			Yield folding records as the file is read.
		streamCSV
			Yield nonempty CSV records as strings.
		streamCSVintegers
			Yield nonempty CSV records as integer tuples.
"""
from __future__ import annotations

from contextlib import suppress
from csv import reader as csv_reader, writer as csv_writer
from datetime import datetime, timedelta, UTC
from email.utils import format_datetime
from hunterMakesPy import errorL33T, oneIndexed
from hunterMakesPy.filesystemToolkit import writeStringToHere
from itertools import repeat, tee
from mapFolding.theSSOT import settingsPackage
from operator import itemgetter
from pathlib import Path, PurePosixPath
from platformdirs import user_data_dir
from sys import modules as sysModules, stdout
from typing import TYPE_CHECKING
from urllib3 import PoolManager
from urllib3.exceptions import HTTPError
import os

if TYPE_CHECKING:
	from _csv import Writer
	from collections.abc import Iterable, Mapping, Sequence
	from io import TextIOWrapper
	from mapFolding.dataBaskets import StateElimination
	from mapFolding.theTypes import Folding, 形Triangle, 形TriangleSparse
	from os import PathLike
	from pandas import DataFrame
	from typing import Any
	from urllib3.response import BaseHTTPResponse

#======== Paths and filenames ======================================================================

def getPathRootJobDEFAULT() -> Path:
	"""Create and return the default directory for computation results.

	(AI generated docstring)

	You can use this function to obtain a writable job-directory location using the package name from
	`mapFolding.theSSOT.settingsPackage` [1]. This function creates the selected directory and missing
	parent directories.

	Directory Selection
	-------------------
	When `'google.colab'` is present in the imported-module registry, the directory is
	`'/content/drive/MyDrive'` followed by the package name. Otherwise, `platformdirs.user_data_dir`
	[2] supplies the user-data directory with the author directory disabled. This function does not
	mount a drive or verify that a Colab drive is mounted.

	Returns
	-------
	pathJobDEFAULT : Path
		Default job directory.

	References
	----------
	[1] `mapFolding.theSSOT.settingsPackage`

	[2] platformdirs directory API.
		https://platformdirs.readthedocs.io/en/latest/api.html
	"""
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / settingsPackage.identifierPackage
	else:
		pathJobDEFAULT = Path(user_data_dir(appname=settingsPackage.identifierPackage, appauthor=False, ensure_exists=True))
	pathJobDEFAULT.mkdir(parents=True, exist_ok=True)
	return pathJobDEFAULT

def makeFilenameArrayFoldings(totalDimensions: int, suffix: str = '.pkl') -> str:
	"""Build the sampled-folding filename for `totalDimensions`.

	(AI generated docstring)

	You can use this function to name sampled folding data for a map with length two in every
	dimension. `makePathFilenameArrayFoldings` [1] combines this filename with a sample-data directory.

	Parameters
	----------
	totalDimensions : int
		Dimension count encoded in the filename.
	suffix : str = '.pkl'
		Suffix appended to the filename stem.

	Returns
	-------
	filenameArrayFoldings : str
		`f'arrayFoldings2上{totalDimensions}Dimensional{suffix}'`.

	References
	----------
	[1] `makePathFilenameArrayFoldings`

	"""
	return makeFilenameCount(f'arrayFoldings2上{totalDimensions}Dimensional', suffix=suffix)

def makeFilenameCount(*underscore: str, suffix: str = '.countTotal', **dash: str) -> str:
	"""Build a count filename from positional and named components.

	(AI generated docstring)

	You can use this function to join computation labels into a filename and append `suffix`.
	This function preserves the supplied component order and does not sanitize characters.

	Parameters
	----------
	*underscore : str
		Positional components joined with underscores.
	suffix : str = '.countTotal'
		Suffix appended after all filename components.
	**dash : str
		Named components added after the positional components. Each key and value is joined
		with a dash, and successive components are joined with underscores.

	Returns
	-------
	filename : str
		Joined components followed by `suffix`. With no components, the result is `suffix`.

	Examples
	--------
	The notebook `easyRun/notebookCount.ipynb` names output from the family, index, and algorithm.

		```python
		pathLikeWrite = getPathRootJobDEFAULT() / makeFilenameCount(kind, str(n), flow)
		```
	"""
	stem: str = '_'.join([*underscore, *map('-'.join, dash.items())])
	return stem + suffix

def makeFilenameFolds(mapShape: tuple[int, ...], suffix: str = '.totalFolds') -> str:
	"""Encode the dimensions in `mapShape` in a folding-count filename.

	(AI generated docstring)

	You can use this function to name an output count from the dimension lengths. This function
	preserves the supplied order; this function does not sort or validate dimensions.

	Parameters
	----------
	mapShape : tuple[int, ...]
		Dimensions to encode, in their supplied order.
	suffix : str = '.totalFolds'
		Suffix appended to the dimension-based stem.

	Returns
	-------
	filenameTotalFolds : str
		The letter `'p'` followed by dimension lengths joined with `'x'`, followed by `suffix`.
		An empty `mapShape` produces `'p'` followed by `suffix`.
	"""
	return makeFilenameCount('p' + 'x'.join(map(str, mapShape)), suffix=suffix)

def makePathFilenameArrayFoldings(
	totalDimensions: int, pathRoot: PathLike[str] = settingsPackage.pathDataSamples, *, suffix: str = '.pkl'
) -> Path:
	"""Locate sampled-folding data for `totalDimensions` under `pathRoot`.

	(AI generated docstring)

	You can use this function to combine a sample-data directory with the filename produced by
	`makeFilenameArrayFoldings` [1]. This function returns a path without creating directories.

	Parameters
	----------
	totalDimensions : int
		Dimension count encoded in the filename.
	pathRoot : PathLike[str] = settingsPackage.pathDataSamples
		Root directory for sampled-folding data. The default is captured from the shared package
		settings [2] when this module is imported.
	suffix : str = '.pkl'
		Suffix appended to the filename stem.

	Returns
	-------
	pathFilenameArrayFoldings : Path
		Sample-data path under `pathRoot`, preserving a relative root when supplied.

	References
	----------
	[1] `makeFilenameArrayFoldings`

	[2] `mapFolding.theSSOT.settingsPackage`

	"""
	return Path(pathRoot) / makeFilenameArrayFoldings(totalDimensions, suffix=suffix)

# TODO Create a truly generic, but useful, make path filename function.
def makePathFilenameCount(
	pathLikeWrite: PathLike[str] | None = None, *underscore: str, suffix: str = '.countTotal', **dash: str
) -> Path:
	"""Resolve a count-output path and create missing parent directories.

	(AI generated docstring)

	You can use this function to choose an output filename from `pathLikeWrite` and computation
	labels. Existing directories receive a filename from `makeFilenameCount` [1]; `None` selects
	the default directory from `getPathRootJobDEFAULT` [2].

	Path Interpretation
	-------------------
	Only an existing directory is treated as a directory. A nonexistent directory-like path is
	treated as a filename. An existing absolute file is returned at its existing location.
	This function does not test file-write permissions.

	Parameters
	----------
	pathLikeWrite : PathLike[str] | None = None
		Existing directory, filename, or file path. A relative nondirectory path is placed under
		the default job directory. An absolute file path keeps its location. An existing relative
		directory receives the generated filename and remains relative.
	*underscore : str
		Positional filename components.
	suffix : str = ".countTotal"
		Suffix used by the generated filename.
	**dash : str
		Named filename components joined as key-value pairs.

	Returns
	-------
	pathFilename : Path
		Resolved output path. Missing parent directories are created.

	References
	----------
	[1] `makeFilenameCount`

	[2] `getPathRootJobDEFAULT`

	"""
	filename: str = makeFilenameCount(*underscore, suffix=suffix, **dash)
	if pathLikeWrite is None:
		pathFilename: Path = getPathRootJobDEFAULT() / filename
	else:
		pathLikeWrite = Path(pathLikeWrite)
		if pathLikeWrite.is_dir():
			pathFilename = pathLikeWrite / filename
		elif pathLikeWrite.is_file() and pathLikeWrite.is_absolute():
			pathFilename = pathLikeWrite
		else:
			pathFilename = getPathRootJobDEFAULT() / pathLikeWrite
		pathFilename.parent.mkdir(parents=True, exist_ok=True)
	return pathFilename

def makePathFilenameFolds(
	mapShape: tuple[int, ...] = (), pathLikeWrite: PathLike[str] | None = None, *, suffix: str = '.totalFolds'
) -> Path:
	"""Resolve a folding-count output path named from `mapShape`.

	(AI generated docstring)

	You can use this function to combine the filename from `makeFilenameFolds` [1] with an output
	location. Path interpretation and directory creation follow `makePathFilenameCount` [2].

	Parameters
	----------
	mapShape : tuple[int, ...] = ()
		Dimensions encoded in the generated filename, preserving their order.
	pathLikeWrite : PathLike[str] | None = None
		Output file or existing directory. `None` selects the default job directory.
	suffix : str = '.totalFolds'
		Suffix appended to the generated filename.

	Returns
	-------
	pathFilenameTotalFolds : Path
		Output path after creating missing parent directories. An existing relative directory can
		produce a relative result.

	References
	----------
	[1] `makeFilenameFolds`

	[2] `makePathFilenameCount`

	"""
	filename: str = makeFilenameFolds(mapShape, suffix)
	return makePathFilenameCount(pathLikeWrite, filename.removesuffix(suffix), suffix=suffix)

#======== Confirm persistence ======================================================================

def saveTotalFAILearly[形PathLike: PathLike[str]](pathFilename: 形PathLike) -> 形PathLike:
	"""Test a new output file before beginning a long computation.

	(AI generated docstring)

	You can use this function to verify that `pathFilename` can be written and read before counting.
	This function rejects an existing path, writes a test integer, reads the integer back, and returns
	the original path object. The test file remains in place.

	Test File Lifetime
	------------------
	This function writes through `hunterMakesPy.filesystemToolkit.writeStringToHere` [1].
	A later call to `saveTotal` [2] normally replaces the test integer with the computed count.
	This function does not reserve the path against concurrent writers or remove the test file
	when subsequent computation fails.

	Parameters
	----------
	pathFilename : 形PathLike
		New output file with an existing parent directory.

	Returns
	-------
	pathFilenameValidated : 形PathLike
		The original path object after a successful write and read.

	Raises
	------
	FileExistsError
		If `pathFilename` already exists.
	FileNotFoundError
		If the parent directory is absent, the test file is absent after writing, or the
		integer read back differs from the test integer.

	References
	----------
	[1] hunterMakesPy filesystem toolkit.
		https://context7.com/hunterhogan/huntermakespy
	[2] `saveTotal`

	"""
	if Path(pathFilename).exists():
		message: str = f"`{pathFilename = }` exists: a battle of overwriting might cause tears."
		raise FileExistsError(message)
	if not Path(pathFilename).parent.exists():
		message = f"I received `{pathFilename = }` 0.000139 seconds ago from a function that promised it created the parent directory, but the parent directory does not exist. Fix that now, so your computation doesn't get deleted later. And be compassionate to others."
		raise FileNotFoundError(message)
	countTotal: int = errorL33T
	writeStringToHere(str(countTotal), pathFilename)
	if not Path(pathFilename).exists():
		message = f"I just wrote a test file to `{pathFilename = }`, but it does not exist. Fix that now, so your computation doesn't get deleted later. And continually improve your empathy skills."
		raise FileNotFoundError(message)
	countTotalRead = int(Path(pathFilename).read_text(encoding="utf-8"))
	if countTotalRead != countTotal:
		message = f"I wrote a test file to `{pathFilename = }` with contents of `{str(countTotal) = }`, but I read `{countTotalRead = }` from the file. Python says the values are not equal. Fix that now, so your computation doesn't get corrupted later. And be pro-social."
		raise FileNotFoundError(message)

	return pathFilename

#======== Save =====================================================================================

def appendTriangle(triangle: 形Triangle, pathFilename: Path) -> Path:
	"""Append the numbered groups in `triangle` to a CSV file.

	(AI generated docstring)

	This function delegates to `writeTriangle` [1] with appending enabled. Existing group keys
	are not checked, so a file can contain repeated groups after appending.

	Parameters
	----------
	triangle : 形Triangle
		Group keys mapped to dense integer sequences.
	pathFilename : Path
		Destination CSV file, created with its parent directories when absent.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied path after appending completes.

	References
	----------
	[1] `writeTriangle`
	"""
	return writeTriangle(triangle, pathFilename, append=True)

def iterableToCSV(boxOfBoxes: Iterable[Iterable[Any]], pathFilename: Path, *, append: bool = False) -> Path:
	"""Write the records in `boxOfBoxes` to `pathFilename`.

	(AI generated docstring)

	You can use this function to save groups as UTF-8 comma-separated records using the default
	Python `csv.writer` dialect [1]. This function creates missing parent directories and returns
	the supplied path after closing the file.

	Parameters
	----------
	boxOfBoxes : Iterable[Iterable[Any]]
		Iterable of group-like records accepted by the CSV writer.
	pathFilename : Path
		Destination file.
	append : bool = False
		Append records when `True`; otherwise replace existing file contents.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied `pathFilename`.

	Examples
	--------
	`writeAlbum` [2] writes each folding as a CSV record.

		```python
		return iterableToCSV(album, pathFilename)
		```

	References
	----------
	[1] Python `csv.writer`.
		https://docs.python.org/3/library/csv.html#csv.writer
	[2] `writeAlbum`

	"""
	pathFilename.parent.mkdir(parents=True, exist_ok=True)
	with pathFilename.open(encoding="utf-8", mode="a" if append else "w", newline="") as streamWrite:
		csvWriter: Writer = csv_writer(streamWrite)
		csvWriter.writerows(boxOfBoxes)
	return pathFilename

def saveTotal(pathFilename: PathLike[str], countTotal: int) -> PurePosixPath:
	"""Save `countTotal` with fallback file output and printed recovery output.

	(AI generated docstring)

	You can use this function to persist a completed integer count. This function first writes to
	`pathFilename` using `hunterMakesPy.filesystemToolkit.writeStringToHere` [1]. If that attempt
	fails, this function prints the count and attempts a file in the working directory.

	Fallback Output
	---------------
	The primary attempt catches `BaseException`, including interruption exceptions. The fallback
	filename depends on the remainder of `countTotal` divided by three and is not unique. The
	fallback can overwrite an earlier fallback file. If the fallback fails, this function attempts
	to print the count again. A failure of that final output operation can still propagate.

	Parameters
	----------
	pathFilename : PathLike[str]
		Preferred output file.
	countTotal : int
		Completed count to write as decimal text.

	Returns
	-------
	pathFilenameWritten : PurePosixPath
		Path representation of the successful output location. If the fallback also fails,
		the result is `PurePosixPath('')`, which renders as `'.'`.

	References
	----------
	[1] hunterMakesPy filesystem toolkit.
		https://context7.com/hunterhogan/huntermakespy
	"""
	try:
		pathFilenameWritten: Path | str = writeStringToHere(str(countTotal), pathFilename)
	#ruff: ignore[blind-except] #=Sin= The total must print.
	except BaseException as ERRORmessage:
		#ruff: ignore[too-many-statements-in-try-clause] #=Sin= Is it a sin to ignore a pretentious rule?
		try:
			stdout.write((banner := '\n' + ' '.join(['countTotal'] * 5) + '\n') + f"\n{countTotal = }\n" + banner)
			stdout.writelines(str(ERRORmessage))
			stdout.write(banner + f"\n{countTotal = }\n" + banner)
			#ruff: ignore[os-getcwd, os-path-join] #=Sin= `try` used pathlib, so I'm using builtin here.
			pathFilenameWritten = os.path.join(os.getcwd(), 'countTotal' + ''.join(((countTotal % 3) + 2) * ['YO_']) + '.txt')
			#ruff: ignore[builtin-open, open-file-with-context-handler] #=Sin= `try` used pathlib and context handler.
			streamWriteFallback: TextIOWrapper = open(pathFilenameWritten, 'w', encoding='utf-8')
			streamWriteFallback.write(str(countTotal))
			streamWriteFallback.close()
			stdout.write(pathFilenameWritten)
		#ruff: ignore[bare-except] #=Sin= The total must print.
		except:
			stdout.write(str(countTotal))
			pathFilenameWritten = ''
	return PurePosixPath(pathFilenameWritten)

def saveTriangleGroup(
	keyGroup: int, boxOfIntegers: Sequence[int], nameTriangle: str, pathLikeWrite: PathLike[str] | None = None
) -> Path:
	"""Save one group of `boxOfIntegers` as a numbered triangle CSV record.

	(AI generated docstring)

	You can use this function to keep the result of one row calculation. This function resolves
	the output path with `makePathFilenameCount` [1], then writes the group with `writeTriangle` [2].

	Parameters
	----------
	keyGroup : int
		Group number written as the first CSV field.
	boxOfIntegers : Sequence[int]
		Dense group values in column order.
	nameTriangle : str
		Triangle name used with the group number in a generated filename.
	pathLikeWrite : PathLike[str] | None = None
		Existing directory, explicit filename, or `None` for the default output location.

	Returns
	-------
	pathFilenameWritten : Path
		Resolved path of the saved group. Existing contents are replaced.

	References
	----------
	[1] `makePathFilenameCount`

	[2] `writeTriangle`
	"""
	return writeTriangle({keyGroup: boxOfIntegers}, makePathFilenameCount(pathLikeWrite, nameTriangle, str(keyGroup), suffix='.csv'))

def writeAlbum(album: Iterable[Folding], pathFilename: Path) -> Path:
	"""Write each folding in `album` as one CSV record.

	(AI generated docstring)

	You can use this function to persist a collection of foldings for `readAlbum` [1] or
	`streamAlbum` [2]. This function delegates writing to `iterableToCSV` [3], which creates missing
	parent directories and replaces existing file contents.

	Parameters
	----------
	album : Iterable[Folding]
		Foldings in their supplied order, with each tuple containing leaf labels by pile position.
	pathFilename : Path
		Destination CSV file.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied path after all records have been written.

	References
	----------
	[1] `readAlbum`

	[2] `streamAlbum`

	[3] `iterableToCSV`

	"""
	return iterableToCSV(album, pathFilename)

def writeDiagonal(lookupDiagonal: Mapping[int, int], pathFilename: Path, 次diagonal: int, *, append: bool = False) -> Path:
	"""Write `lookupDiagonal` as sorted group, diagonal-offset, and value records.

	(AI generated docstring)

	You can use this function to save one diagonal in the three-column format accepted by
	`readDiagonal` [1]. Writing uses `iterableToCSV` [2].

	CSV Layout
	----------
	Records contain the group key, `次diagonal`, and value, in ascending group-key order with
	no header. Appending duplicate group keys for the same diagonal produces a file that the
	diagonal parser rejects.

	Parameters
	----------
	lookupDiagonal : Mapping[int, int]
		Group numbers mapped to the stored diagonal values.
	pathFilename : Path
		Destination CSV file.
	次diagonal : int
		Diagonal offset repeated in every output record. This function does not validate the value.
	append : bool = False
		Append records when `True`; otherwise replace existing file contents.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied path after writing completes.

	Examples
	--------
	The runner in `easyRun/triangleSemiMake.py` appends a completed count from
	`mapFolding.dataBaskets.StateMeanders.totalMeanders` [3] to a diagonal file.

		```python
		writeDiagonal(
			{n: state.totalMeanders},
			makePathFilenameCount(settingsPackage.pathPackage / 'oeis' / 'A400429', 'diagonal', str(次diagonal), suffix='.csv'),
			次diagonal, append=True
		)
		```

	References
	----------
	[1] `readDiagonal`

	[2] `iterableToCSV`

	[3] `mapFolding.dataBaskets.StateMeanders.totalMeanders`

	"""
	keysGroups, keysContents = tee(sorted(lookupDiagonal))
	return iterableToCSV(
		zip(keysGroups, repeat(次diagonal, len(lookupDiagonal)), map(lookupDiagonal.__getitem__, keysContents), strict=True)
		, pathFilename, append=append
	)

def writeText(contents: str, pathFilename: Path) -> Path:
	"""Write `contents` to `pathFilename` and create missing parent directories.

	(AI generated docstring)

	This function creates the parent directory tree and delegates writing to
	`hunterMakesPy.filesystemToolkit.writeStringToHere` [1]. Existing contents are replaced.

	Parameters
	----------
	contents : str
		Text to write.
	pathFilename : Path
		Destination file.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied path after writing completes.

	References
	----------
	[1] hunterMakesPy filesystem toolkit.
		https://github.com/hunterhogan/hunterMakesPy
	"""
	writeStringToHere(contents, pathFilename)
	return pathFilename

def writeTriangle(triangle: 形Triangle | 形TriangleSparse, pathFilename: Path, *, append: bool = False) -> Path:
	"""Save numbered integer groups from `triangle` to a CSV file.

	(AI generated docstring)

	You can use this function to persist a triangle for later loading with `readTriangle` [1] or
	`readTriangleSparse` [2]. This function writes the groups to `pathFilename` and returns
	`pathFilename` after closing the file.

	CSV Layout
	----------
	Each record contains the group key followed by that group's values, without a header. Records
	appear in ascending group-key order. Dense groups retain their supplied cell order, and empty
	dense groups contain only the group key. Sparse groups use column keys starting at one and
	write empty fields for missing cells, extending through at least the group key or the largest
	stored column key. `mapFolding.dataStructures.getBoxWithGaps` [3] supplies this sparse layout.

	The file uses UTF-8 and the default `csv.writer` dialect [4]. `iterableToCSV` [5] creates
	missing parent directories and replaces existing contents unless `append=True`. An empty
	`triangle` produces an empty file when replacing, or appends no records when appending.
	Group lengths are not validated.

	Parameters
	----------
	triangle : 形Triangle | 形TriangleSparse
		Group keys mapped to dense integer collections or sparse integer lookups.
	pathFilename : Path
		Destination file. Existing contents are overwritten unless `append=True`.
	append : bool = False
		Append numbered groups when `True`.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied `pathFilename` after writing completes.

	Writing Failures
	----------------
	This function propagates `OSError` if creating parent directories or opening or writing the
	file fails.

	Examples
	--------
	The writer in `easyRun/reduceArches.py` saves accumulated groups using its existing `triangle`
	and destination directory `pathWrite`.

		```python
		from mapFolding.kitFilesystem import writeTriangle

		writeTriangle(triangle, pathWrite / 'A287548ADDENDUM.csv')
		```

	References
	----------
	[1] `readTriangle`

	[2] `readTriangleSparse`

	[3] `mapFolding.dataStructures.getBoxWithGaps`

	[4] csv.writer - Python standard library.
		https://docs.python.org/3/library/csv.html#csv.writer
	[5] `iterableToCSV`

	"""
	from mapFolding.dataStructures import getBoxWithGaps
	return iterableToCSV(
		map(lambda keyGroup: (keyGroup, *getBoxWithGaps(triangle[keyGroup], keyGroup)), sorted(triangle))  # ruff: ignore[unnecessary-map]
		, pathFilename, append=append
	)

def writeTriangleSparse(triangleSparse: 形TriangleSparse, pathFilename: Path) -> Path:
	"""Save `triangleSparse` as numbered CSV groups with empty fields for missing columns.

	(AI generated docstring)

	This function uses the sparse layout of `writeTriangle` [1] and replaces existing contents.
	Read the result with `readTriangleSparse` [2] to preserve the original column keys.

	Parameters
	----------
	triangleSparse : 形TriangleSparse
		Group keys mapped to sparse cell lookups with positive column keys.
	pathFilename : Path
		Destination CSV file.

	Returns
	-------
	pathFilenameWritten : Path
		The supplied path after writing completes.

	References
	----------
	[1] `writeTriangle`

	[2] `readTriangleSparse`
	"""
	return writeTriangle(triangleSparse, pathFilename)

#======== Read and save ===========================================================================

def getCacheOrURL(pathFilenameCache: Path, cacheDays: int, url: str) -> str:
	"""Read cached text or request updated text from `url`.

	(AI generated docstring)

	You can use this function to retrieve UTF-8 text while reusing `pathFilenameCache`. A recent
	cache returns immediately. A stale or missing cache triggers an HTTP request with urllib3 [1].
	Suppressed request failures leave the previously read text unchanged.

	Cache Refresh
	-------------
	A stale cache adds the HTTP `If-Modified-Since` header [2]. Status 304 updates the cache
	modification time; status 200 decodes the response and writes the new text using
	`hunterMakesPy.filesystemToolkit.writeStringToHere` [3]. Other statuses retain the earlier text.
	The request disables retries. This function suppresses urllib3 `HTTPError` and `AttributeError`,
	but filesystem and text-decoding errors can propagate.

	Parameters
	----------
	pathFilenameCache : Path
		Local cache file.
	cacheDays : int
		Cache lifetime in days, measured from the file modification time.
	url : str
		Source URL. The protocol prefix is checked only when a request is needed.

	Returns
	-------
	contents : str
		Cached or retrieved text. A missing cache and an unsuccessful request produce an empty string.

	Raises
	------
	ValueError
		If a request is needed and `url` does not start with `'http:'` or `'https:'`.

	References
	----------
	[1] urllib3 documentation.
		https://urllib3.readthedocs.io/en/stable/
	[2] RFC 9110, HTTP `If-Modified-Since`.
		https://www.rfc-editor.org/rfc/rfc9110.html#name-if-modified-since
	[3] hunterMakesPy filesystem toolkit.
		https://context7.com/hunterhogan/huntermakespy
	"""
	preferCache: bool = False
	contents: str = ''
	cacheDatetime: datetime | None = None

	if pathFilenameCache.exists():
		cacheDatetime = datetime.fromtimestamp(pathFilenameCache.stat().st_mtime, tz=UTC)
		preferCache = datetime.now(tz=UTC) - cacheDatetime < timedelta(days=cacheDays)
		contents = pathFilenameCache.read_text(encoding="utf-8")

	if not preferCache:
		if not url.startswith(("http:", "https:")):
			message: str = f"I received {url = }, but it must start with 'http:' or 'https:'"
			raise ValueError(message)

		headers: dict[str, str] | None = None
		if cacheDatetime is not None:
			headers = {"If-Modified-Since": format_datetime(cacheDatetime, usegmt=True)}

		httpPoolManager = PoolManager(retries=False)
		with suppress(HTTPError, AttributeError):
			response: BaseHTTPResponse = httpPoolManager.request("GET", url, headers=headers, preload_content=True, decode_content=True)
			if response.status == 304:  # Not Modified
				pathFilenameCache.touch()  # Update cache file's modification time to local time.
			elif response.status == 200:
				writeStringToHere(contents := response.data.decode("utf-8"), pathFilenameCache)
		httpPoolManager.clear()

	return contents

#======== Read =====================================================================================

def getDataFrameFoldings(state: StateElimination) -> DataFrame:
	"""Load bundled array-folding data for the dimension count in `state`.

	(AI generated docstring)

	You can use this function to obtain sample data for an elimination state. This function resolves
	the standard sample-data path with `makePathFilenameArrayFoldings` [1] and loads the file with
	`readDataFrame` [2]. Loading errors propagate to the caller.

	Parameters
	----------
	state : StateElimination
		State providing `state.totalDimensions`.

	Returns
	-------
	dataframeFoldings : DataFrame
		Sample data as a pandas dataframe [3].

	See Also
	--------
	`readDataFrame`
		Load a dataframe from an explicitly chosen file.

	References
	----------
	[1] `makePathFilenameArrayFoldings`

	[2] `readDataFrame`

	[3] pandas `DataFrame`.
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
	"""
	return readDataFrame(makePathFilenameArrayFoldings(state.totalDimensions))

def readAlbum(pathFilename: Path) -> tuple[Folding, ...]:
	"""Load all foldings from `pathFilename` into a tuple.

	(AI generated docstring)

	You can use this function to read an album saved by `writeAlbum` [1]. This function converts each
	nonempty CSV record [2] to a tuple of integers and loads every record into memory.

	Parameters
	----------
	pathFilename : Path
		UTF-8 album file without a header.

	Returns
	-------
	album : tuple[Folding, ...]
		Foldings in file order. An empty file returns an empty tuple.

	See Also
	--------
	`streamAlbum`
		Iterate over folding records as the file is read.

	References
	----------
	[1] `writeAlbum`

	[2] Python `csv.reader`.
		https://docs.python.org/3/library/csv.html#csv.reader
	"""
	return tuple(streamAlbum(pathFilename))

def readDataFrame(pathFilename: PathLike[str]) -> DataFrame:
	"""Load serialized data from `pathFilename` as a dataframe.

	(AI generated docstring)

	You can use this function to load stored folding data into a pandas dataframe [1].
	This function reads the file with `pandas.read_pickle` [2] and passes the loaded object to the
	dataframe constructor. Loading and construction errors propagate to the caller.

	Parameters
	----------
	pathFilename : PathLike[str]
		File containing data serialized for the pickle reader.

	Returns
	-------
	dataframeFoldings : DataFrame
		Dataframe constructed from the loaded object.

	See Also
	--------
	`getDataFrameFoldings`
		Load sample data using the dimension count in an elimination state.

	References
	----------
	[1] pandas `DataFrame`.
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
	[2] pandas `read_pickle`.
		https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html
	"""
	#=Sin= `pandas` is optional.
	#ruff: ignore[import-outside-top-level]
	import pandas
	return pandas.DataFrame(pandas.read_pickle(pathFilename))

def readDiagonal(pathFilename: Path, 次diagonal: int) -> dict[int, int]:
	"""Read diagonal `次diagonal` from three-field integer CSV records.

	(AI generated docstring)

	You can use this function to load counts saved by `writeDiagonal` [1]. Each nonempty record
	contains a group key, a diagonal offset, and a value. This function selects records with the
	requested offset and returns the values in ascending group-key order.

	Parameters
	----------
	pathFilename : Path
		UTF-8 CSV file with three integer fields per nonempty record and no header.
	次diagonal : int
		Diagonal offset to select. This function does not validate the offset range.

	Returns
	-------
	lookupDiagonal : dict[int, int]
		Selected values keyed by group number, or an empty dictionary when no record matches.

	Raises
	------
	ValueError
		If any nonempty record has a field count other than three, a field cannot be parsed as
		an integer, or selected records repeat a group key. Malformed unselected records also fail.

	References
	----------
	[1] `writeDiagonal`
	"""
	#=SIN= Intermediate tuple: duplicate-row validation needs the selected record count after dictionary construction.
	boxOfBoxes: tuple[tuple[int, ...], ...] = tuple(streamCSVintegers(pathFilename))
	if not all(map(lambda boxOfIntegers: len(boxOfIntegers) == 3, boxOfBoxes)):  # ruff: ignore[unnecessary-map]
		message: str = 'I received a diagonal CSV entry without exactly three integers: a group key, a diagonal offset, and an integer.'
		raise ValueError(message)
	boxOfBoxes = tuple(filter(lambda boxOfIntegers: boxOfIntegers[1] == 次diagonal, boxOfBoxes))
	lookupDiagonal: dict[int, int] = dict(map(itemgetter(0, 2), boxOfBoxes))
	if len(lookupDiagonal) != len(boxOfBoxes):
		message = f"I received duplicate group keys for `{次diagonal = }`."
		raise ValueError(message)
	return dict(sorted(lookupDiagonal.items()))

def readText(pathFilename: Path, *, default: str | None = None) -> str:
	"""Read the contents of `pathFilename` as UTF-8 text.

	(AI generated docstring)

	You can use this function to load a text file into one string. `default` supplies a value
	when the file is absent. Other file-access and decoding errors propagate to the caller.

	Parameters
	----------
	pathFilename : Path
		Text file to read.
	default : str | None = None
		Return this text when the file is absent; `None` propagates `FileNotFoundError`.

	Returns
	-------
	contents : str
		Decoded file contents.

	Examples
	--------
	`mapFolding.oeis.updateBFile` [1] reads an existing b-file or starts with empty contents.

		```python
		contents = readText(pathFilename, default='')
		```

	References
	----------
	[1] `mapFolding.oeis.updateBFile`

	"""
	try:
		return pathFilename.read_text(encoding='utf-8')
	except FileNotFoundError:
		if default is None:
			# raise from for context?
			raise
		return default

def readTriangle(pathFilename: Path) -> dict[int, tuple[int, ...]]:
	"""Load numbered integer groups from a triangle CSV file.

	(AI generated docstring)

	You can use this function to load groups previously saved by `writeTriangle` [1]. This function
	streams the records from `pathFilename` and returns a dictionary mapping group keys to tuples
	of values.

	CSV Layout
	----------
	The file must contain UTF-8 text with no header. Each CSV record [2] begins with an integer group
	number followed by integer values. `streamCSVintegers` [3] ignores empty
	records and preserves the supplied group order. A record containing only a group key produces an
	empty tuple. Repeated group keys keep the last values at the first occurrence's dictionary
	position [4]. An empty file produces an empty dictionary. Group lengths and consecutive numbering
	are not validated.

	Parameters
	----------
	pathFilename : Path
		Triangle file to read.

	Returns
	-------
	triangle : dict[int, tuple[int, ...]]
		Group numbers mapped to all values after each group key.

	Reading Failures
	----------------
	This function propagates `OSError` if opening or reading `pathFilename` fails, and
	`UnicodeDecodeError` if the file does not contain valid UTF-8 text. A field that cannot be
	converted to an integer, including a header or empty field, propagates `ValueError` from
	the parser.

	References
	----------
	[1] `writeTriangle`

	[2] csv.reader - Python standard library.
		https://docs.python.org/3/library/csv.html#csv.reader
	[3] `streamCSVintegers`

	[4] Dictionary insertion order - Python standard library.
		https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
	"""
	return dict(map(itemgetter(0, slice(oneIndexed, None)), streamCSVintegers(pathFilename)))

def readTriangleSparse(pathFilename: Path) -> dict[int, dict[int, int]]:
	"""Load numbered sparse groups from `pathFilename` without shifting missing columns.

	(AI generated docstring)

	You can use this function to recover sparse cells saved by `writeTriangleSparse` [1] or
	`writeTriangle` [2]. Empty cell fields are omitted while later cells retain their original
	column keys. `mapFolding.dataStructures.stringsToLookup` [3] converts the cell fields.

	Parameters
	----------
	pathFilename : Path
		UTF-8 CSV file without a header. Each nonempty record starts with an integer group key,
		followed by integer cell fields or empty fields.

	Returns
	-------
	triangleSparse : dict[int, dict[int, int]]
		Groups mapped to available values with column keys starting at one. Empty records are
		ignored. Repeated group keys retain the last record's cells, and a record containing only
		a group key produces an empty dictionary.

	See Also
	--------
	`readTriangle`
		Read dense integer records when every cell field contains an integer.

	References
	----------
	[1] `writeTriangleSparse`

	[2] `writeTriangle`

	[3] `mapFolding.dataStructures.stringsToLookup`
	"""
	from mapFolding.dataStructures import getBoxWithGaps, stringsToLookup
	streamKeys, streamGroups = tee(streamCSV(pathFilename))
	return dict(zip(
		map(int, map(itemgetter(0), streamKeys))
		, map(stringsToLookup, map(itemgetter(slice(oneIndexed, None)), streamGroups))
		, strict=True
	))

#======== Stream ===================================================================================

def streamAlbum(pathFilename: Path) -> Iterable[Folding]:
	"""Yield foldings from `pathFilename` as the records are read.

	(AI generated docstring)

	You can use this function to process an album saved by `writeAlbum` [1] without loading all
	records into memory. Each nonempty CSV record [2] becomes a tuple of integers.

	Open File Lifetime
	------------------
	The file is opened when iteration starts and remains open until the generator finishes or is
	closed. Retain and close the generator if iteration stops before exhausting the file.

	Parameters
	----------
	pathFilename : Path
		UTF-8 album file without a header.

	Yields
	------
	folding : Folding
		One folding in file order.

	See Also
	--------
	`readAlbum`
		Load all folding records into a tuple.

	References
	----------
	[1] `writeAlbum`

	[2] Python `csv.reader`.
		https://docs.python.org/3/library/csv.html#csv.reader
	"""
	yield from streamCSVintegers(pathFilename)

def streamCSV(pathFilename: Path) -> Iterable[list[str]]:
	"""Yield nonempty CSV records from `pathFilename` as lists of strings.

	(AI generated docstring)

	This function reads UTF-8 records using the default Python `csv.reader` dialect [1]. Empty
	records are skipped; records containing empty fields are retained. Headers and comments
	receive no special treatment.

	Parameters
	----------
	pathFilename : Path
		CSV file to read.

	Yields
	------
	boxOfStrings : list[str]
		One nonempty record in file order, preserving each field as parsed text.

	Open File Lifetime
	------------------
	The file opens when iteration begins and remains open until the generator finishes or is
	closed. Close the generator if iteration ends before the file is exhausted.

	References
	----------
	[1] Python `csv.reader`.
		https://docs.python.org/3/library/csv.html#csv.reader
	"""
	with pathFilename.open(encoding='utf-8', mode='r', newline='') as streamRead:
		yield from filter(bool, csv_reader(streamRead))

def streamCSVintegers(pathFilename: Path) -> Iterable[tuple[int, ...]]:
	"""Yield nonempty CSV records from `pathFilename` as integer tuples.

	(AI generated docstring)

	This function converts every field read by `streamCSV` [1] using `int`. Headers, comments,
	and empty fields are not skipped and must be absent for conversion to succeed.

	Parameters
	----------
	pathFilename : Path
		UTF-8 CSV file containing integer fields.

	Yields
	------
	boxOfIntegers : tuple[int, ...]
		One nonempty record in file order.

	References
	----------
	[1] `streamCSV`
	"""
	yield from map(lambda boxOfStrings: tuple(map(int, boxOfStrings)), streamCSV(pathFilename))  # ruff: ignore[unnecessary-map]
