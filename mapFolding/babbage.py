import multiprocessing
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union

from Z0Z_tools import defineConcurrencyLimit

from .distributionCenter import getIndicesRemaining, pathTasksToParameters
from .lovelace import foldings
from .prepareParameters import countMinimumParsePoints, getDimensions


def computeSeries(series: str, X_n: int, normalFoldings: bool = True) -> int:
    return computeSeriesConcurrently(series, X_n, 1, normalFoldings)

def computeSeriesConcurrently(series: str, X_n: int, CPUlimit: Optional[Union[int, float, bool]] = None, normalFoldings: bool = True) -> int:
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

def computeSeriesTask(dimensions: list[int], computationDivisions: int, computationIndex: int, normalFoldings: bool) -> int:
    return foldings(dimensions, computationDivisions, computationIndex, normalFoldings)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
