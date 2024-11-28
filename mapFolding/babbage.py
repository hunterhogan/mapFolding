from .lovelace import foldings
from .prepareParameters import getDimensions, countMinimumParsePoints
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union
from Z0Z_tools import defineConcurrencyLimit
import multiprocessing
import os
import pathlib
import random

def computeSeries(series: str, X_n: int, normalFoldings: bool = True) -> int:
    return computeSeriesConcurrently(series, X_n, normalFoldings, 1)

def computeSeriesConcurrently(series: str, X_n: int, normalFoldings: bool = True, CPUlimit: Optional[Union[int, float, bool]] = None) -> int:
    max_workers = defineConcurrencyLimit(CPUlimit)
    count = 0
    if X_n == 0:
        return 1
    else:
        dimensions = getDimensions(series, X_n)
    computationDivisions = countMinimumParsePoints(dimensions)
    with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
        listConcurrency = [concurrencyManager.submit(computeSeriesTask, dimensions, normalFoldings, computationIndex, computationDivisions)
                           for computationIndex in range(computationDivisions)]
        listCountsFolding = [concurrency.result() for concurrency in listConcurrency]
    count = sum(listCountsFolding)
    return count

def computeSeriesTask(dimensions: list[int], normalFoldings: bool, computationIndex: int, computationDivisions: int) -> int:
    return foldings(dimensions, computationDivisions, computationIndex, normalFoldings)

def computeDistributedTask(pathTasks: Union[str, os.PathLike[str]], CPUlimit: Optional[Union[int, float, bool]] = None) -> None:
    max_workers = defineConcurrencyLimit(CPUlimit)
    series, X_n, normalFoldings, computationDivisions = pathlib.PurePosixPath(pathTasks).parts[-4:]
    dimensions = getDimensions(series, int(X_n))
    listComputationIndices = list(range(int(computationDivisions)))
    for indexCompleted in pathlib.Path(pathTasks).glob('*'):
        listComputationIndices.remove(int(indexCompleted.stem))
    random.shuffle(listComputationIndices)
    listIndicesTasks = listComputationIndices[0:max_workers]
    with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
        dictionaryConcurrency = {concurrencyManager.submit(
            computeSeriesTask, dimensions, bool(normalFoldings), computationIndex, int(computationDivisions))
                                 : computationIndex for computationIndex in listIndicesTasks}
        for claimTicket in as_completed(dictionaryConcurrency):
            indexCompleted = dictionaryConcurrency[claimTicket]
            pathlib.Path(pathTasks, str(indexCompleted)).write_text(str(claimTicket.result()))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
