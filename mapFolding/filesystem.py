"""Filesystem functions for mapFolding package."""
from pathlib import Path, PurePath
from typing import Any
import os

def getFilenameFoldsTotal(mapShape: tuple[int, ...]) -> str:
	"""Imagine your computer has been counting folds for 9 days, and when it tries to save your newly discovered value,
	the filename is invalid. I bet you think this function is more important after that thought experiment.

	Make a standardized filename for the computed value `foldsTotal`.

	The filename takes into account
		- the dimensions of the map, aka `mapShape`, aka `listDimensions`
		- no spaces in the filename
		- safe filesystem characters
		- unique extension
		- Python-safe strings:
			- no starting with a number
			- no reserved words
			- no dashes or other special characters
			- uh, I can't remember, but I found some other frustrating limitations
		- if 'p' is still the first character of the filename, I picked that because it was the original identifier for the map shape in Lunnan's code

	Parameters:
		mapShape: A sequence of integers representing the dimensions of the map.

	Returns:
		filenameFoldsTotal: A filename string in format 'pMxN.foldsTotal' where M,N are sorted dimensions
	"""
	return 'p' + 'x'.join(str(dimension) for dimension in sorted(mapShape)) + '.foldsTotal'

def getPathFilenameFoldsTotal(mapShape: tuple[int, ...], pathLikeWriteFoldsTotal: str | os.PathLike[str] | None = None) -> Path:
	"""Get a standardized path and filename for the computed value `foldsTotal`.

	If you provide a directory, the function will append a standardized filename. If you provide a filename
	or a relative path and filename, the function will prepend the default path.

	Parameters:
		mapShape: List of dimensions for the map folding problem.
		pathLikeWriteFoldsTotal (pathJobRootDEFAULT): Path, filename, or relative path and filename. If None, uses default path.
			Defaults to None.

	Returns:
		pathFilenameFoldsTotal: Absolute path and filename.
	"""
	from mapFolding.theSSOT import getPathJobRootDEFAULT

	if pathLikeWriteFoldsTotal is None:
		pathFilenameFoldsTotal = getPathJobRootDEFAULT() / getFilenameFoldsTotal(mapShape)
	else:
		pathLikeSherpa = Path(pathLikeWriteFoldsTotal)
		if pathLikeSherpa.is_dir():
			pathFilenameFoldsTotal = pathLikeSherpa / getFilenameFoldsTotal(mapShape)
		elif pathLikeSherpa.is_file() and pathLikeSherpa.is_absolute():
			pathFilenameFoldsTotal = pathLikeSherpa
		else:
			pathFilenameFoldsTotal = getPathJobRootDEFAULT() / pathLikeSherpa

	pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
	return pathFilenameFoldsTotal

def saveFoldsTotal(pathFilename: str | os.PathLike[str], foldsTotal: int) -> None:
	"""
	Save foldsTotal with multiple fallback mechanisms.

	Parameters:
		pathFilename: Target save location
		foldsTotal: Critical computed value to save
	"""
	try:
		pathFilenameFoldsTotal = Path(pathFilename)
		pathFilenameFoldsTotal.parent.mkdir(parents=True, exist_ok=True)
		pathFilenameFoldsTotal.write_text(str(foldsTotal))
	except Exception as ERRORmessage:
		try:
			print(f"\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n\n{foldsTotal=}\n\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n")
			print(ERRORmessage)
			print(f"\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n\n{foldsTotal=}\n\nfoldsTotal foldsTotal foldsTotal foldsTotal foldsTotal\n")
			randomnessPlanB = (int(str(foldsTotal).strip()[-1]) + 1) * ['YO_']
			filenameInfixUnique = ''.join(randomnessPlanB)
			pathFilenamePlanB = os.path.join(os.getcwd(), 'foldsTotal' + filenameInfixUnique + '.txt')
			writeStreamFallback = open(pathFilenamePlanB, 'w')
			writeStreamFallback.write(str(foldsTotal))
			writeStreamFallback.close()
			print(str(pathFilenamePlanB))
		except Exception:
			print(foldsTotal)
	return None

def writeStringToHere(this: str, pathFilename: str | os.PathLike[Any] | PurePath) -> None:
	"""Write the string `this` to the file at `pathFilename`."""
	pathFilename = Path(pathFilename)
	pathFilename.parent.mkdir(parents=True, exist_ok=True)
	pathFilename.write_text(str(this))
	return None
