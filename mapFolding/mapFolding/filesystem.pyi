import os
from collections.abc import Sequence as Sequence
from numpy import dtype as dtype, integer as integer, ndarray as ndarray
from pathlib import Path

def saveFoldsTotal(pathFilename: str | os.PathLike[str], foldsTotal: int) -> None:
    """
\tSave foldsTotal with multiple fallback mechanisms.

\tParameters:
\t\tpathFilename: Target save location
\t\tfoldsTotal: Critical computed value to save
\t"""
def getFilenameFoldsTotal(mapShape: tuple[int, ...]) -> str:
    """Imagine your computer has been counting folds for 9 days, and when it tries to save your newly discovered value,
\tthe filename is invalid. I bet you think this function is more important after that thought experiment.

\tMake a standardized filename for the computed value `foldsTotal`.

\tThe filename takes into account
\t\t- the dimensions of the map, aka `mapShape`, aka `listDimensions`
\t\t- no spaces in the filename
\t\t- safe filesystem characters
\t\t- unique extension
\t\t- Python-safe strings:
\t\t\t- no starting with a number
\t\t\t- no reserved words
\t\t\t- no dashes or other special characters
\t\t\t- uh, I can't remember, but I found some other frustrating limitations
\t\t- if 'p' is still the first character of the filename, I picked that because it was the original identifier for the map shape in Lunnan's code

\tParameters:
\t\tmapShape: A sequence of integers representing the dimensions of the map.

\tReturns:
\t\tfilenameFoldsTotal: A filename string in format 'pMxN.foldsTotal' where M,N are sorted dimensions
\t"""
def getPathFilenameFoldsTotal(mapShape: tuple[int, ...], pathLikeWriteFoldsTotal: str | os.PathLike[str] | None = None) -> Path:
    """Get a standardized path and filename for the computed value `foldsTotal`.

\tIf you provide a directory, the function will append a standardized filename. If you provide a filename
\tor a relative path and filename, the function will prepend the default path.

\tParameters:
\t\tmapShape: List of dimensions for the map folding problem.
\t\tpathLikeWriteFoldsTotal (pathJobRootDEFAULT): Path, filename, or relative path and filename. If None, uses default path.
\t\t\tDefaults to None.

\tReturns:
\t\tpathFilenameFoldsTotal: Absolute path and filename.
\t"""
