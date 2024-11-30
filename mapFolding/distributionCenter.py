import os
import pathlib
import random
from typing import Optional, Union


def getIndicesRemaining(pathTasks: Union[str, os.PathLike[str]], computationDivisions: int) -> list[int]:
    """
    Get the computation indices that have not been written to disk.

    Parameters:
        pathTasks: The path to the directory containing the computation index files.
        computationDivisions: The total number of computation divisions.

    Returns:
        listComputationIndices: A shuffled list of remaining computation indices that have
        not been completed.
    """
    listComputationIndices = list(range(int(computationDivisions)))
    for indexCompleted in pathlib.Path(pathTasks).glob('*.computationIndex'):
        listComputationIndices.remove(int(indexCompleted.stem))
    random.shuffle(listComputationIndices)
    return listComputationIndices

def sumDistributedTasks(pathTasks: Union[str, os.PathLike[str]]) -> Optional[int]:
    """
    Sum the results of distributed tasks.

    Parameters:
        pathTasks: The path to the directory containing the computation index files.

    Returns:
        sumFoldingCounts: The total number of foldings.
    """
    series, X_n, computationDivisions, normalFoldings = pathTasksToParameters(pathTasks)
    listIndicesRemaining = getIndicesRemaining(pathTasks, computationDivisions)
    if listIndicesRemaining:
        print(f"Warning: {len(listIndicesRemaining)} tasks have not been completed.")
        return None

    sumFoldingCounts = 0
    for index in range(computationDivisions):
        pathFilename = pathlib.Path(pathTasks, f"{str(index)}.computationIndex")
        allegedInteger = pathFilename.read_text().strip()
        if not allegedInteger.isdigit():
            print(f"Warning: {pathFilename} does not contain an integer.")
            return None
        sumFoldingCounts += int(allegedInteger)
    return sumFoldingCounts

def pathTasksToParameters(pathTasks: Union[str, os.PathLike[str]]) -> tuple[str, int, int, bool]:
    """
    Extracts the parameters from the pathTasks string.
    
    Parameters:
        pathTasks: A string containing the path to the task file.
    
    Returns:
        series,X_n,computationDivisions,normalFoldings: 
        The series type of the map, e.g. '2', '3', '2 X 2', 'n';
        The number of dimensions, n, for the specified series;
        The number of divisions to make in the computation;
        When True, enumerate only normal foldings.
    """
    series, X_n, computationDivisions, normalFoldings = pathlib.PurePosixPath(pathTasks).parts[-4:]
    return series, int(X_n), int(computationDivisions), bool(normalFoldings)

# function: given the same parameters as `computeSeries` and a path
# - figure out the subpath for distributed computation
# - create the subpath if necessary
# - return the the subpath as a value
# Maybe - optionally create a file with the path in it, such as how I am using 
# `pathTasks = pathlib.Path(pathlib.Path("/content/drive/MyDrive/dataHunter/mapFolding/pathTasks.txt").read_text())`