# ruff: file-ignore[suspicious-pickle-usage]
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
makeFilenameFoldsTotal, makePathFilenameFoldsTotal
	Variants tuned for foldsTotal outputs.
saveFoldsTotal, saveFoldsTotalFAILearly
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

from hunterMakesPy import errorL33T
from hunterMakesPy.filesystemToolkit import writeStringToHere
from mapFolding import ansiColorReset, ansiColors
from mapFolding.theSSOT import settingsPackage
from pathlib import Path, PurePosixPath
from sys import modules as sysModules, stdout
from typing import TYPE_CHECKING
import csv
import os
import pandas
import platformdirs
import sys

if TYPE_CHECKING:
	from _csv import Writer
	from collections.abc import Iterable, Iterator
	from io import TextIOWrapper
	from mapFolding._e.dataBaskets import EliminationState
	from mapFolding._e.theTypes import Folding
	from os import PathLike

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
		pathJobDEFAULT = Path(platformdirs.user_data_dir(appname=settingsPackage.identifierPackage, appauthor=False, ensure_exists=True))
	pathJobDEFAULT.mkdir(parents=True, exist_ok=True)
	return pathJobDEFAULT

def makePathFilenameCount(pathLikeWrite: PathLike[str] | None = None, *underscore: str, suffix: str = ".countTotal", **dash: str) -> Path:
	"""Create an absolute `pathlib.Path` for a 'countTotal' filename.

	(AI generated docstring)

	Parameters
	----------
	pathLikeWrite : os.PathLike | None = None
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

def makePathFilenameFolds(mapShape: tuple[int, ...] = (), pathLikeWrite: PathLike[str] | None = None, *, suffix: str = '.foldsTotal') -> Path:
	"""Get a standardized filename and create a configurable path to store the computed `foldsTotal` value.

	To help reduce duplicate code and to increase predictability, this function creates a standardized
	filename, has a default but configurable path, and creates the path.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A sequence of integers representing the map dimensions.
	pathLikeWrite : PathLike[str] | None = getPathRootJobDEFAULT()
		Path, filename, or relative path and filename. If None, uses default path. If a directory,
		appends standardized filename.
	suffix : str = '.foldsTotal'
		Filename suffix/extension to use.

	Returns
	-------
	pathFilenameFoldsTotal : Path
		Absolute path and filename for storing the `foldsTotal` value.

	Notes
	-----
	The function creates any necessary directories in the path if they don't exist.
	"""
	filename: str = makeFilenameFolds(mapShape, suffix)
	return makePathFilenameCount(pathLikeWrite, filename.removesuffix(suffix), suffix=suffix)

def makeFilenameFolds(mapShape: tuple[int, ...], suffix: str = '.foldsTotal') -> str:
	"""Create a standardized filename for a computed `foldsTotal` value.

	(AI generated docstring)

	This function generates a consistent, filesystem-safe filename based on map dimensions.
	Standardizing filenames ensures that results can be reliably stored and retrieved, avoiding
	potential filesystem incompatibilities or Python naming restrictions.

	Parameters
	----------
	mapShape : tuple[int, ...]
		A sequence of integers representing the dimensions of the map.
	suffix : str = '.foldsTotal'
		Filename suffix/extension to use.

	Returns
	-------
	filenameFoldsTotal : str
		A filename string in format 'pMxN.foldsTotal' where M,N are sorted dimensions.

	Notes
	-----
	The filename format ensures no spaces in the filename, safe filesystem characters, unique
	extension (.foldsTotal), Python-safe strings (no starting with numbers, no reserved words), and
	the 'p' prefix comes from Lunnon's original code.

	"""
	return makeFilenameCount('p' + 'x'.join(map(str, mapShape)), suffix=suffix)

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
	except Exception as ERRORmessage:  # ruff:ignore[blind-except]
		try:  # ruff:ignore[too-many-statements-in-try-clause]
			stdout.write((banner := '\n' + ' '.join(['countTotal'] * 5) + '\n') + f"\n{countTotal = }\n" + banner)
			stdout.writelines(str(ERRORmessage))
			stdout.write(banner + f"\n{countTotal = }\n" + banner)
			pathFilenameWritten = os.path.join(os.getcwd(), 'countTotal' + ''.join(((countTotal % 3) + 2) * ['YO_']) + '.txt')  # ruff:ignore[os-getcwd, os-path-join]
			streamWriteFallback: TextIOWrapper = open(pathFilenameWritten, 'w', encoding='utf-8')  # ruff:ignore[builtin-open, open-file-with-context-handler]
			streamWriteFallback.write(str(countTotal))
			streamWriteFallback.close()
			stdout.write(pathFilenameWritten)
		except Exception:  # ruff:ignore[blind-except]
			stdout.write(str(countTotal))
			pathFilenameWritten = ''
	return PurePosixPath(pathFilenameWritten)

def writeAlbum(album: Iterable[Folding], pathFilename: Path) -> Path:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	with pathFilename.open(encoding="utf-8", mode="w", newline="", buffering=2**16) as streamWrite:
		csvWriter: Writer = csv.writer(streamWrite)
		csvWriter.writerows(album)
	return pathFilename

#================== Read ==========================================================================

# TODO generalize `getDataFrameFoldings`.
def getDataFrameFoldings(state: EliminationState) -> pandas.DataFrame | None:  # ruff: ignore[undocumented-public-function]
	pathFilename: Path = Path(f'{settingsPackage.pathPackage}/tests/dataSamples/arrayFoldingsP2d{state.dimensionsTotal}.pkl')
	dataframeFoldings: pandas.DataFrame | None = None
	if pathFilename.exists():
		dataframeFoldings = pandas.DataFrame(pandas.read_pickle(pathFilename))
	else:
		message: str = f"{ansiColors.YellowOnBlack}I received {state.dimensionsTotal = }, but I could not find the data at:\n\t{pathFilename!r}.{ansiColorReset}"
		sys.stderr.write(message + '\n')
	return dataframeFoldings

def readAlbum(pathFilename: Path) -> tuple[Folding, ...]:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	with pathFilename.open(encoding="utf-8", mode="r", newline="") as streamRead:
		return tuple(tuple(map(int, row)) for row in csv.reader(streamRead))

def streamAlbum(pathFilename: Path) -> Iterable[Folding]:  # ruff: ignore[undocumented-public-function]
	# DOCUMENT
	with pathFilename.open(encoding="utf-8", mode="r", newline="") as streamRead:
		csvReader: Iterator[list[str]] = csv.reader(streamRead)
		for row in csvReader:
			yield tuple(map(int, row))

# Perhaps:
#================== Find or enumerate files based on their purpose, not filename or path ==========
