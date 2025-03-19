"""
Filesystem utilities for managing map folding computation results.

This module provides functions for standardized handling of files related to the mapFolding
package, with a focus on saving, retrieving, and naming computation results. It implements
consistent naming conventions and path resolution strategies to ensure that:

1. Computation results are stored in a predictable location
2. Filenames follow a consistent pattern based on map dimensions
3. Results can be reliably retrieved for future reference
4. The system handles file operations safely with appropriate error handling

The module serves as the interface between the computational components of the package
and the filesystem, abstracting away the details of file operations and path management.
"""
from pathlib import Path, PurePath
from typing import Any
from os import PathLike
import os

def getFilenameFoldsTotal(mapShape: tuple[int, ...]) -> str:
    """
    Create a standardized filename for a computed `foldsTotal` value.

    This function generates a consistent, filesystem-safe filename based on map dimensions.
    Standardizing filenames ensures that results can be reliably stored and retrieved,
    avoiding potential filesystem incompatibilities or Python naming restrictions.

    Parameters:
        mapShape: A sequence of integers representing the dimensions of the map.

    Returns:
        filenameFoldsTotal: A filename string in format 'pMxN.foldsTotal' where M,N are sorted dimensions.

    Notes:
        The filename format ensures:
        - No spaces in the filename
        - Safe filesystem characters
        - Unique extension (.foldsTotal)
        - Python-safe strings (no starting with numbers, no reserved words)
        - The 'p' prefix preserves compatibility with Lunnan's original code.
    """
    return 'p' + 'x'.join(str(dimension) for dimension in sorted(mapShape)) + '.foldsTotal'

def getPathFilenameFoldsTotal(mapShape: tuple[int, ...], pathLikeWriteFoldsTotal: str | PathLike[str] | None = None) -> Path:
    """
    Get a standardized path and filename for the computed foldsTotal value.

    This function resolves paths for storing computation results, handling different
    input types including directories, absolute paths, or relative paths. It ensures
    that all parent directories exist in the resulting path.

    Parameters:
        mapShape: List of dimensions for the map folding problem.
        pathLikeWriteFoldsTotal (getPathJobRootDEFAULT): Path, filename, or relative path and filename.
            If None, uses default path. If a directory, appends standardized filename.

    Returns:
        pathFilenameFoldsTotal: Absolute path and filename for storing the foldsTotal value.

    Notes:
        The function creates any necessary directories in the path if they don't exist.
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

def saveFoldsTotal(pathFilename: str | PathLike[str], foldsTotal: int) -> None:
    """
    Save `foldsTotal` value to disk with multiple fallback mechanisms.

    This function attempts to save the computed `foldsTotal` value to the specified
    location, with backup strategies in case the primary save attempt fails.
    The robustness is critical since these computations may take days to complete.

    Parameters:
        pathFilename: Target save location for the `foldsTotal` value
        foldsTotal: The computed value to save

    Notes:
        If the primary save fails, the function will attempt alternative save methods:
        1. Print the value prominently to stdout
        2. Create a fallback file in the current working directory
        3. As a last resort, simply print the value
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

def writeStringToHere(this: str, pathFilename: str | PathLike[Any] | PurePath) -> None:
    """
    Write a string to a file, creating parent directories if needed.

    Parameters:
        this: The string content to write to the file
        pathFilename: The target file path where the string should be written
    """
    pathFilename = Path(pathFilename)
    pathFilename.parent.mkdir(parents=True, exist_ok=True)
    pathFilename.write_text(str(this))
    return None
