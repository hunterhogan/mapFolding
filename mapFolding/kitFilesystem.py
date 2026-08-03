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
"""

from __future__ import annotations

from hunterMakesPy import errorL33T
from mapFolding.theSSOT import settingsPackage
from pathlib import Path
from sys import modules as sysModules, stdout
from typing import TYPE_CHECKING
import os
import platformdirs

if TYPE_CHECKING:
	from io import TextIOWrapper
	from os import PathLike

def makePathFilenameCountTotal(pathLikeWrite: PathLike[str] | None = None, *underscore: str, suffix: str = ".countTotal", **dash: str) -> Path:
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
	filename: str = makeFilenameCountTotal(*underscore, suffix=suffix, **dash)
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

def makeFilenameCountTotal(*underscore: str, suffix: str = '.countTotal', **dash: str) -> str:
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

def makePathFilenameFoldsTotal(mapShape: tuple[int, ...] = (), pathLikeWrite: PathLike[str] | None = None, *, suffix: str = '.foldsTotal') -> Path:
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
	filename: str = makeFilenameFoldsTotal(mapShape, suffix)
	return makePathFilenameCountTotal(pathLikeWrite, filename.removesuffix(suffix), suffix=suffix)

def makeFilenameFoldsTotal(mapShape: tuple[int, ...], suffix: str = '.foldsTotal') -> str:
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
	return makeFilenameCountTotal('p' + 'x'.join(map(str, mapShape)), suffix=suffix)

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

def _saveFoldsTotal(pathFilename: PathLike[str], foldsTotal: int) -> None:
	"""Save a `foldsTotal` value to a file.

	(AI generated docstring)

	This function provides the core file writing functionality used by the public `saveFoldsTotal`
	function. It handles the basic operations of creating parent directories and writing the integer
	value as text to the specified file location.

	Parameters
	----------
	pathFilename : PathLike[str]
		Path where the `foldsTotal` value should be saved.
	foldsTotal : int
		The integer value to save.

	Notes
	-----
	This is an internal function that doesn't include error handling or fallback mechanisms. Use
	`saveFoldsTotal` for production code that requires robust error handling.

	"""
	pathFilenameFoldsTotal = Path(pathFilename)
	pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
	pathFilenameFoldsTotal.write_text(str(foldsTotal), encoding='utf-8')

def saveFoldsTotal(pathFilename: PathLike[str], foldsTotal: int) -> None:
	"""Save `foldsTotal` value to disk with multiple fallback mechanisms.

	(AI generated docstring)

	This function attempts to save the computed `foldsTotal` value to the specified location, with
	backup strategies in case the primary save attempt fails. The robustness is critical since these
	computations may take days to complete.

	Parameters
	----------
	pathFilename : PathLike[str]
		Target save location for the `foldsTotal` value.
	foldsTotal : int
		The computed value to save.

	Notes
	-----
	If the primary save fails, the function will attempt alternative save methods. Print the value
	prominently to `stdout`. Create a fallback file in the current working directory. As a last
	resort, simply print the value.

	The fallback filename includes a unique identifier based on the value itself to prevent conflicts.

	"""
	try:
		_saveFoldsTotal(pathFilename, foldsTotal)
	except Exception as ERRORmessage:  # ruff:ignore[blind-except]
		try:  # ruff:ignore[too-many-statements-in-try-clause]
			stdout.write(f"\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n\n{foldsTotal = }\n\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n")
			stdout.writelines(str(ERRORmessage))
			stdout.write(f"\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n\n{foldsTotal = }\n\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n")
			randomnessPlanB: list[str] = (int(str(foldsTotal).strip()[-1]) + 1) * ['YO_']
			filenameInfixUnique: str = ''.join(randomnessPlanB)
			pathFilenamePlanB: str = os.path.join(os.getcwd(), 'foldsTotal' + filenameInfixUnique + '.txt')  # ruff:ignore[os-getcwd, os-path-join]
			writeStreamFallback: TextIOWrapper = open(pathFilenamePlanB, 'w', encoding='utf-8')  # ruff:ignore[builtin-open, open-file-with-context-handler]
			writeStreamFallback.write(str(foldsTotal))
			writeStreamFallback.close()
			stdout.write(str(pathFilenamePlanB))
		except Exception:  # ruff:ignore[blind-except]
			stdout.write(str(foldsTotal))

def saveFoldsTotalFAILearly(pathFilename: PathLike[str]) -> None:
	"""Preemptively test file write capabilities before beginning computation.

	(AI generated docstring)

	This function performs validation checks on the target file location before a potentially
	long-running computation begins. It tests several critical aspects of filesystem functionality to
	ensure results can be saved.

	Parameters
	----------
	pathFilename : PathLike[str]
		The path and filename where computation results will be saved.

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
	_saveFoldsTotal(pathFilename, countTotal)
	if not Path(pathFilename).exists():
		message = f"I just wrote a test file to `{pathFilename = }`, but it does not exist. Fix that now, so your computation doesn't get deleted later. And continually improve your empathy skills."
		raise FileNotFoundError(message)
	countTotalRead = int(Path(pathFilename).read_text(encoding="utf-8"))
	if countTotalRead != countTotal:
		message = f"I wrote a test file to `{pathFilename = }` with contents of `{str(countTotal) = }`, but I read `{countTotalRead = }` from the file. Python says the values are not equal. Fix that now, so your computation doesn't get corrupted later. And be pro-social."
		raise FileNotFoundError(message)
