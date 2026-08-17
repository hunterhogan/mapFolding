#ruff: file-ignore[suspicious-pickle-usage] #=Sin= Centralize all pickle usage here.
"""Persistent storage utilities for map folding computation results.

(AI generated docstring)

This module provides helpers to generate standardized filenames and paths for storing results produced
by long-running map folding computations. It centralizes logic for platform-aware default locations,
filename construction, safe path creation, and robust save/fallback behavior.

Primary utilities
-----------------
makeFilenameCountTotal
	Build a filesystem-safe filename for countTotal-style outputs.
makePathFilenameCountTotal
	Resolve or create a Path for a given filename or directory and ensure parent directories exist.
makeFilenameTotalFolds, makePathFilenameTotalFolds
	Variants tuned for totalFolds outputs.
saveTotalFolds, saveTotalFoldsFAILearly
	Functions that write computed results to disk with fallback and validation.

See Also
--------
hunterMakesPy.filesystemToolkit.writeStringToHere
hunterMakesPy.filesystemToolkit.importLogicalPath2Identifier
hunterMakesPy.filesystemToolkit.importPathFilename2Identifier
hunterMakesPy.filesystemToolkit.writePython

astToolkit
- reading
- writing
- containers with methods
"""
from __future__ import annotations

from contextlib import suppress
from csv import reader as csv_reader, writer as csv_writer
from datetime import datetime, timedelta, UTC
from email.utils import format_datetime
from hunterMakesPy import errorL33T
from hunterMakesPy.filesystemToolkit import writeStringToHere
from mapFolding import ansiColorReset, ansiColors
from mapFolding.theSSOT import settingsPackage
from pathlib import Path, PurePosixPath
from platformdirs import user_data_dir
from sys import modules as sysModules, stdout
from typing import TYPE_CHECKING
from urllib3 import PoolManager
from urllib3.exceptions import HTTPError
import os
import sys

if TYPE_CHECKING:
	from _csv import Writer
	from collections.abc import Iterable, Iterator
	from io import TextIOWrapper
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import Folding
	from os import PathLike
	from pandas import DataFrame
	from urllib3.response import BaseHTTPResponse

#================== Create appropriate paths and filenames =========================================

def getPathRootJobDEFAULT() -> Path:
	"""Get the default root directory for map folding computation jobs.

	(AI generated docstring)

	This function determines the appropriate default directory for storing computation results based
	on the current runtime environment. It uses platform-specific directories for normal environments
	and adapts to special environments like Google Colab.

	Returns
	-------
	pathJobDEFAULT : Path
		Path to the default directory for storing computation results.

	Notes
	-----
	For standard environments, uses `platformdirs` to find appropriate user data directory. For Google
	Colab, uses a specific path in Google Drive. Creates the directory if it doesn't exist.

	"""
	if 'google.colab' in sysModules:
		pathJobDEFAULT: Path = Path("/content/drive/MyDrive") / settingsPackage.identifierPackage
	else:
		pathJobDEFAULT = Path(user_data_dir(appname=settingsPackage.identifierPackage, appauthor=False, ensure_exists=True))
	pathJobDEFAULT.mkdir(parents=True, exist_ok=True)
	return pathJobDEFAULT

def makeFilenameArrayFoldings(totalDimensions: int, suffix: str = '.pkl') -> str:
	"""Build the standard filename for array-foldings data.

	(AI generated docstring)

	You can use this function when you want the package's preferred filename for array-foldings data
	with `totalDimensions`. This function keeps the filename convention in one place, so callers do
	not need to know or repeat that convention themselves.

	Parameters
	----------
	totalDimensions : int
		Total number of dimensions represented by the array-foldings data.
	suffix : str = '.pkl'
		Filename suffix appended to the standard filename stem.

	Returns
	-------
	filenameArrayFoldings : str
		Standard filename for the array-foldings data.
	"""
	return makeFilenameCount(f'arrayFoldings2上{totalDimensions}Dimensional', suffix=suffix)

def makeFilenameCount(*underscore: str, suffix: str = '.countTotal', **dash: str) -> str:
	"""Build a standardized filename for countTotal-like outputs.

	(AI generated docstring)

	Parameters
	----------
	*underscore : str
		Positional segments to include in the filename stem; joined with underscores.
	suffix : str = ".countTotal"
		Filename suffix/extension to use.
	**dash : str
		Keyword segments included via ``dash.items()``; each (key, value) is joined with ``'-'`` and
		included in the stem.

	Returns
	-------
	filename : str
		The generated filename string (stem + suffix).
	"""
	stem: str = '_'.join([*underscore, *map('-'.join, dash.items())])
	return stem + suffix

def makeFilenameFolds(mapShape: tuple[int, ...], suffix: str = '.totalFolds') -> str:
	"""Create a standardized filename for a computed `totalFolds` value.

	(AI generated docstring)

	This function generates a consistent, filesystem-safe filename based on map dimensions.
	Standardizing filenames ensures that results can be reliably stored and retrieved, avoiding
	potential filesystem incompatibilities or Python naming restrictions.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A sequence of integers representing the dimensions of the map.
	suffix : str = '.totalFolds'
		Filename suffix/extension to use.

	Returns
	-------
	filenameTotalFolds : str
		A filename string in format 'pMxN.totalFolds' where M,N are sorted dimensions.

	Notes
	-----
	The filename format ensures no spaces in the filename, safe filesystem characters, unique
	extension (.totalFolds), Python-safe strings (no starting with numbers, no reserved words), and
	the 'p' prefix comes from Lunnon's original code.

	"""
	return makeFilenameCount('p' + 'x'.join(map(str, mapShape)), suffix=suffix)

def makePathFilenameArrayFoldings(totalDimensions: int, pathRoot: PathLike[str] = settingsPackage.pathDataSamples, *, suffix: str = '.pkl') -> Path:
	"""Build the standard path for array-foldings data.

	(AI generated docstring)

	You can use this function when you want the package's preferred path for array-foldings data with
	`totalDimensions`. This function combines `pathRoot` with the standard filename and returns the
	result as a `Path`. This function does not create directories.

	Parameters
	----------
	totalDimensions : int
		Total number of dimensions represented by the array-foldings data.
	pathRoot : PathLike[str] = settingsPackage.pathDataSamples
		Root directory that should contain the array-foldings data.
	suffix : str = '.pkl'
		Filename suffix appended to the standard filename stem.

	Returns
	-------
	pathFilenameArrayFoldings : Path
		Standard path for the array-foldings data under `pathRoot`.
	"""
	return Path(pathRoot) / makeFilenameArrayFoldings(totalDimensions, suffix=suffix)

def makePathFilenameCount(pathLikeWrite: PathLike[str] | None = None, *underscore: str, suffix: str = ".countTotal", **dash: str) -> Path:
	"""Create an absolute `pathlib.Path` for a 'countTotal' filename.

	(AI generated docstring)

	Parameters
	----------
	pathLikeWrite : PathLike[str] | None = None
		Directory, filename, or relative path to use. If ``None``, the default job directory returned
		by `getPathRootJobDEFAULT` is used. If a directory is provided, the standardized
		filename is appended. If an absolute file path is provided, it is returned as-is.
	*underscore : str
		Positional segments used to build the filename stem; segments are joined with underscores.
	suffix : str = ".countTotal"
		Filename suffix/extension to use.
	**dash : str
		Keyword segments included via ``dash.items()``; each (key, value) pair is joined with ``'-'``
		and included in the stem; all parts are joined by ``'_'``.

	Returns
	-------
	pathFilename : pathlib.Path
		Absolute path to the filename. Parent directories are created if necessary.
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

def makePathFilenameFolds(mapShape: tuple[int, ...] = (), pathLikeWrite: PathLike[str] | None = None, *, suffix: str = '.totalFolds') -> Path:
	"""Get a standardized filename and create a configurable path to store the computed `totalFolds` value.

	To help reduce duplicate code and to increase predictability, this function creates a standardized
	filename, has a default but configurable path, and creates the path.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A sequence of integers representing the map dimensions.
	pathLikeWrite : PathLike[str] | None = getPathRootJobDEFAULT()
		Path, filename, or relative path and filename. If None, uses default path. If a directory,
		appends standardized filename.
	suffix : str = '.totalFolds'
		Filename suffix/extension to use.

	Returns
	-------
	pathFilenameTotalFolds : Path
		Absolute path and filename for storing the `totalFolds` value.

	Notes
	-----
	The function creates any necessary directories in the path if they don't exist.
	"""
	filename: str = makeFilenameFolds(mapShape, suffix)
	return makePathFilenameCount(pathLikeWrite, filename.removesuffix(suffix), suffix=suffix)

#================== Confirm the ability to read or write ===========================================

def saveTotalFAILearly[形PathLike: PathLike[str]](pathFilename: 形PathLike) -> 形PathLike:
	"""Preemptively test file write capabilities before beginning computation.

	(AI generated docstring)

	This function performs validation checks on the target file location before a potentially
	long-running computation begins. It tests several critical aspects of filesystem functionality to
	ensure results can be saved.

	Parameters
	----------
	pathFilename : PathLike[str]
		The path and filename where computation results will be saved.

	Returns
	-------
	pathFilename : PathLike[str]
		The validated path and filename for saving results.

	Raises
	------
	FileExistsError
		If the target file already exists.
	FileNotFoundError
		If parent directories don't exist or if write tests fail.

	Notes
	-----
	Checks performed: 1. Checks if the file already exists to prevent accidental overwrites. 2.
	Verifies that parent directories exist. 3. Tests if the system can write a test value to the file.
	4. Confirms that the written value can be read back correctly.

	This function helps prevent a situation where a computation runs for hours or days only to
	discover at the end that results cannot be saved. The test value used is a large integer that
	exercises both the writing and reading mechanisms thoroughly.

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

#================== Write =========================================================================

def saveTotal(pathFilename: PathLike[str], countTotal: int) -> PurePosixPath:
	"""Save `countTotal` value to disk with multiple fallback mechanisms.

	(AI generated docstring)

	This function attempts to save the computed `countTotal` value to the specified location, with
	backup strategies in case the primary save attempt fails. The robustness is critical since these
	computations may take days to complete.

	Parameters
	----------
	pathFilename : PathLike[str]
		Target save location for the `countTotal` value.
	countTotal : int
		The computed value to save.

	Returns
	-------
	pathFilenameWritten : PurePosixPath
		The path where the value was successfully saved, or an empty string if all attempts failed.
		`PurePosixPath` because it is easier to persist the `PurePosixPath` object across platforms,
		and it is harder to accidentally modify the value: protect the programmer from themselves. :)

	Notes
	-----
	If the primary save fails, the function will attempt alternative save methods. Print the value
	prominently to `stdout`. Create a fallback file in the current working directory. As a last
	resort, simply print the value of `countTotal`.

	The fallback filename includes a unique identifier based on the `countTotal` value itself to
	prevent conflicts.
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

def writeAlbum(album: Iterable[Folding], pathFilename: Path) -> Path:
	"""Write an album of foldings to a CSV file.

	(AI generated docstring)

	Each `Folding` in the album is written as a row in a CSV file, where each
	element (a `Leaf` index) becomes a cell. The file is opened with a large
	buffer (64 KiB) for efficient I/O during potentially large writes.

	Parameters
	----------
	album : Iterable[Folding]
		An iterable of `Folding` objects to write. Each `Folding` is a tuple of
		integers (leaf indices).
	pathFilename : pathlib.Path
		Destination path for the CSV file. Parent directories are not created
		automatically; ensure they exist before calling.

	Returns
	-------
	pathFilename : pathlib.Path
		The path to the file that was written.

	Notes
	-----
	The CSV format uses no quoting by default (standard `csv.writer` behavior),
	which is safe because `Leaf` values are integers. The large buffer size
	(`2**16` bytes) reduces the number of system calls when writing many rows.
	"""
	with pathFilename.open(encoding="utf-8", mode="w", newline="", buffering=2**16) as streamWrite:
		csvWriter: Writer = csv_writer(streamWrite)
		csvWriter.writerows(album)
	return pathFilename

#================== Read ==========================================================================

def getCacheOrURL(pathFilenameCache: Path, cacheDays: int, url: str) -> str:
	"""I use this to manage cached data retrieval with HTTP conditional requests.

	This caching layer minimizes network traffic by checking local cache validity based on file
	modification time and using HTTP If-Modified-Since headers [1] for efficient updates. The function
	implements a three-tier strategy: prefer valid cache, use conditional HTTP requests with the
	`urllib3` [2] library when cache is stale, fall back to stale cache on network errors.

	Parameters
	----------
	pathFilenameCache : Path
		Path to the local cache file for storing retrieved data.
	cacheDays : int
		Number of days to consider the cache valid.
	url : str
		URL to retrieve the data from if cache is invalid or missing.

	Returns
	-------
	data : str
		The retrieved data as a string.

	References
	----------
	[1] HTTP If-Modified-Since - RFC 9110
		https://www.rfc-editor.org/rfc/rfc9110.html#name-if-modified-since
	[2] urllib3 - Context7
		https://urllib3.readthedocs.io/en/stable/
	"""
	preferCache: bool = False
	data: str = ''
	cacheDatetime: datetime | None = None

	if pathFilenameCache.exists():
		cacheDatetime = datetime.fromtimestamp(pathFilenameCache.stat().st_mtime, tz=UTC)
		preferCache = datetime.now(tz=UTC) - cacheDatetime < timedelta(days=cacheDays)
		data = pathFilenameCache.read_text(encoding="utf-8")

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
			if response.status == 304:
				pathFilenameCache.touch()  # Update cache file's modification time to server time.
			elif response.status == 200:
				writeStringToHere(data := response.data.decode("utf-8"), pathFilenameCache)
		httpPoolManager.clear()

	return data

def getDataFrameFoldings(state: EliminationState) -> DataFrame | None:
	"""Load array-foldings data for `state.totalDimensions`.

	(AI generated docstring)

	You can use this function when you want the package's array-foldings data for an
	`EliminationState`. This function looks in the package's standard data location for
	`state.totalDimensions`, returns the data as a `pandas.DataFrame` [1], and returns `None` after
	writing a diagnostic to the standard error stream when the data is unavailable.

	Parameters
	----------
	state : EliminationState
		Elimination state that supplies `state.totalDimensions`.

	Returns
	-------
	dataframeFoldings : DataFrame | None
		Array-foldings data for `state.totalDimensions`, or `None` when the data is unavailable.

	References
	----------
	[1] pandas.DataFrame
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
	"""
	pathFilename: Path = makePathFilenameArrayFoldings(state.totalDimensions)
	dataframeFoldings: DataFrame | None = None
	if pathFilename.exists():
		dataframeFoldings = readDataFrame(pathFilename)
	else:
		message: str = f"{ansiColors.YellowOnBlack}I received {state.totalDimensions = }, but I could not find the data at:\n\t{pathFilename!r}.{ansiColorReset}"
		sys.stderr.write(message + '\n')
	return dataframeFoldings

def readAlbum(pathFilename: Path) -> tuple[Folding, ...]:
	"""Read an entire album of foldings from a CSV file into memory.

	(AI generated docstring)

	Each row in the CSV file is parsed into a `Folding` (a tuple of integers).
	All rows are materialized into a tuple before returning, so the entire file
	is loaded into memory.

	Parameters
	----------
	pathFilename : pathlib.Path
		Path to a CSV file previously written by `writeAlbum`.

	Returns
	-------
	album : tuple[Folding, ...]
		A tuple of `Folding` objects, one per row in the CSV file. Each
		`Folding` is a tuple of integers (leaf indices).

	See Also
	--------
	streamAlbum : Lazily iterate over foldings without loading the entire file.
	writeAlbum : Write an album of foldings to a CSV file.
	"""
	with pathFilename.open(encoding="utf-8", mode="r", newline="") as streamRead:
		return tuple(tuple(map(int, row)) for row in csv_reader(streamRead))

def readDataFrame(pathFilename: PathLike[str]) -> DataFrame:
	"""Load folding data from `pathFilename` into a `pandas.DataFrame`.

	(AI generated docstring)

	You can use this function when you already know the data file location and want the folding data
	in tabular form. This function keeps the package's DataFrame-loading convention in one place and
	returns the loaded `pandas.DataFrame` [1].

	Parameters
	----------
	pathFilename : PathLike[str]
		Path to the data file.

	Returns
	-------
	dataframeFoldings : DataFrame
		Folding data loaded from `pathFilename` as a `DataFrame`.

	See Also
	--------
	`getDataFrameFoldings`
		Load array-foldings data from the package's standard location.

	References
	----------
	[1] pandas.DataFrame
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
	"""
	#=Sin= `pandas` is optional.
	import pandas  # ruff: ignore[import-outside-top-level]
	return pandas.DataFrame(pandas.read_pickle(pathFilename))

def streamAlbum(pathFilename: Path) -> Iterable[Folding]:
	"""Lazily iterate over foldings in a CSV file, yielding one at a time.

	(AI generated docstring)

	Unlike `readAlbum`, this function does not load the entire file into memory.
	Instead, it opens the file and yields each row as a `Folding` (a tuple of
	integers) as it is read. This is useful for processing large albums without
	consuming excessive memory.

	Parameters
	----------
	pathFilename : pathlib.Path
		Path to a CSV file previously written by `writeAlbum`.

	Yields
	------
	folding : Folding
		Each row from the CSV file, converted to a tuple of integers (leaf
		indices).

	Notes
	-----
	The file remains open for the lifetime of the iterator. If the iterator is
	not fully consumed, the file handle is closed when the generator is
	garbage-collected or explicitly closed.

	See Also
	--------
	readAlbum : Read an entire album into memory at once.
	writeAlbum : Write an album of foldings to a CSV file.
	"""
	with pathFilename.open(encoding="utf-8", mode="r", newline="") as streamRead:
		csvReader: Iterator[list[str]] = csv_reader(streamRead)
		for row in csvReader:
			yield tuple(map(int, row))

# Perhaps:
#================== Find or enumerate files based on their purpose, not filename or path ==========
