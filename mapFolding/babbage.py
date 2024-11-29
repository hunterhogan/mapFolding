import multiprocessing
import os
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union

from Z0Z_tools import defineConcurrencyLimit

from .lovelace import foldings
from .prepareParameters import countMinimumParsePoints, getDimensions, pathTasksToParameters


def computeSeries(series: str, X_n: int, normalFoldings: bool = True) -> int:
    """
    Calculates the number of valid folding sequences for a given series pattern.

    Parameters:
        series: the series type of the map, e.g. '2', '3', '2 X 2', 'n'
        X_n: the number of dimensions, n, for the specified series
        normalFoldings (True): when True, enumerate only normal foldings

    Returns:
        foldingsTotal: The total number of foldings for the series and size specified.
    """
    return computeSeriesConcurrently(series, X_n, 1, normalFoldings)

def computeSeriesConcurrently(series: str, X_n: int, CPUlimit: Optional[Union[int, float, bool]] = None, normalFoldings: bool = True) -> int:
    """
    Parameters:
        CPUlimit (gluttonous resource usage): What limitations, if any, to place on CPU usage.
    """
    
    # I want a better way to generalize "2 X n" and such.
    # TODO: make docstring available without duplicating writing the docstring
    max_workers = defineConcurrencyLimit(CPUlimit)
    if X_n == 0:
        return 1
    else:
        dimensions = getDimensions(series, X_n)
    computationDivisions = countMinimumParsePoints(dimensions)
    with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
        listConcurrency = [concurrencyManager.submit(computeSeriesTask, dimensions, computationDivisions, computationIndex, normalFoldings)
                           for computationIndex in range(computationDivisions)]
        listCountsFolding = [concurrency.result() for concurrency in listConcurrency]
    return sum(listCountsFolding)

def computeDistributedTask(pathTasks: Union[str, os.PathLike[str]], CPUlimit: Optional[Union[int, float, bool]] = None) -> int:
    """
    Computes distributed tasks for map folding calculations. The quantity of counted folds are written directly to a file named by its index in the `pathTasks` directory.

    Parameters:
        pathTasks:
            Path containing task instructions encoded in format
            'series/X_n/computationDivisions/normalFoldings'
        CPUlimit (gluttonous resource usage): CPU core limit for parallel processing.
            See `Z0Z_tools.defineConcurrencyLimit()` for interpretation details.
    Returns:
        COUNTindicesWithoutFile: The number of remaining tasks to be computed.
    """
    max_workers = defineConcurrencyLimit(CPUlimit)

    series, X_n, computationDivisions, normalFoldings = pathTasksToParameters(pathTasks)
    
    dimensions = getDimensions(series, X_n)

    listIndicesTasks = getIndicesRemaining(pathTasks, computationDivisions)[0:max_workers]

    if listIndicesTasks:
        with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
            dictionaryConcurrency = {concurrencyManager.submit(
                computeSeriesTask, dimensions, computationDivisions, computationIndex, normalFoldings)
                                    : computationIndex for computationIndex in listIndicesTasks}

            for claimTicket in as_completed(dictionaryConcurrency):
                indexCompleted = dictionaryConcurrency[claimTicket]
                pathlib.Path(pathTasks, f"{str(indexCompleted)}.computationIndex").write_text(str(claimTicket.result()))

    return len(getIndicesRemaining(pathTasks, computationDivisions))

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

def computeSeriesTask(dimensions: list[int], computationDivisions: int, computationIndex: int, normalFoldings: bool) -> int:
    """Ideally, this is the only link between this module and the algorithm, `foldings()`."""
    return foldings(dimensions, computationDivisions, computationIndex, normalFoldings)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
