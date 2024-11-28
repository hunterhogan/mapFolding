from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union
import multiprocessing
import os
import pathlib
import random

def computeSeries(series: str, X_n: int) -> int:
    return computeSeriesConcurrently(series, X_n, True, 1)

def computeSeriesConcurrently(series: str, X_n: int, normalFoldings: bool = True, CPUlimit: Optional[Union[int, float, bool]] = None) -> int:
    if CPUlimit is None:
        CPUlimit = 0
    max_workers = getCPUlimit(CPUlimit)
    count = 0
    if X_n == 0:
        return 1
    else:
        dimensions = getDimensions(series, X_n)
    computationDivisions = countMinimumParsePoints(dimensions)
    with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
        listConcurrency = [concurrencyManager.submit(computeSeriesTask, dimensions, normalFoldings, computationIndex, computationDivisions)
                           for computationIndex in range(computationDivisions)]
        listPortionValues = [concurrency.result() for concurrency in listConcurrency]
    count = sum(listPortionValues)
    return count

def getCPUlimit(CPUlimit: Union[int, float, bool]) -> int:
    max_workers = multiprocessing.cpu_count()
    if CPUlimit is not None:
        if isinstance(CPUlimit, bool):
            if CPUlimit == True:
                max_workers = 1
        elif isinstance(CPUlimit, int):
            if CPUlimit > 0:
                max_workers = CPUlimit
            elif CPUlimit == 0:
                pass
            elif CPUlimit < 0:
                max_workers = max(multiprocessing.cpu_count() + CPUlimit, 1)
        elif isinstance(CPUlimit, float):
            max_workers = max(int(CPUlimit * multiprocessing.cpu_count()), 1)
    return max_workers

def countMinimumParsePoints(dimensions: list[int]) -> int:
    leavesTotal = 1
    for dimensionSize in dimensions:
        leavesTotal *= dimensionSize
    COUNTreachesParsePoint = sum(1 for potentialDivision in range(1, leavesTotal + 1) 
                                if any(potentialDivision == 1 or potentialDivision - dimensionSize >= 1 
                                      for dimensionSize in dimensions))
    return COUNTreachesParsePoint

def getDimensions(series: str, X_n: int) -> list[int]:  
    if isinstance(series, int):
        series = str(series)
    if series == '2':
        return [2, X_n]
    elif series == '3':
        return [3, X_n]
    elif series.lower() == '2 x 2':
        return [2] * X_n
    elif series == 'n':
        return [X_n + 1, X_n + 1]
    else:
        return [int(series), X_n]

def computeSeriesTask(dimensions: list[int], normalFoldings: bool, computationIndex: int, computationDivisions: int) -> int:
    count_holder = [0]
    foldings(dimensions, normalFoldings, computationIndex, computationDivisions, count_holder)
    return count_holder[0]

def computeDistributedTask(pathTasks: Union[str, os.PathLike[str]], CPUlimit: Optional[Union[int, float, bool]] = None) -> None:
    if CPUlimit is None:
        CPUlimit = 0
    max_workers = getCPUlimit(CPUlimit)
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

def process(a, b, n: int, count_holder) -> None:
    count_holder[0] += n

def foldings(p: list[int], normalFoldings: bool = True, computationIndex: int = 0, computationDivisions: int = 0, count_holder=None) -> None:
    if count_holder is None:
        count_holder = [0]
    n = 1
    for pp in p:
        n *= pp
    a = [0] * (n + 1)
    b = [0] * (n + 1)
    count = [0] * (n + 1)
    gapter = [0] * (n + 1)
    gap = [0] * (n * n + 1)
    dim = len(p)
    bigP = [1] * (dim + 1)
    c = [[0] * (n + 1) for _ in range(dim + 1)]
    d = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(dim + 1)]
    for i in range(1, dim + 1):
        bigP[i] = bigP[i - 1] * p[i - 1]
    for i in range(1, dim + 1):
        for m in range(1, n + 1):
            c[i][m] = (m - 1) // bigP[i - 1] - ((m - 1) // bigP[i]) * p[i - 1] + 1
    for i in range(1, dim + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = c[i][l] - c[i][m]
                if (delta & 1) == 0:
                    d[i][l][m] = m if c[i][m] == 1 else m - bigP[i - 1]
                else:
                    d[i][l][m] = m if c[i][m] == p[i - 1] or m + bigP[i - 1] > l else m + bigP[i - 1]
    g = 0
    l = 1
    while l > 0:
        if not normalFoldings or l <= 1 or b[0] == 1:
            if l > n:
                process(a, b, n, count_holder)
            else:
                dd = 0
                gg = gapter[l - 1]
                g = gg
                for i in range(1, dim + 1):
                    if d[i][l][l] == l:
                        dd += 1
                    else:
                        m = d[i][l][l]
                        while m != l:
                            if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                                gap[gg] = m
                                count[m] += 1
                                gg += 1
                            m = d[i][l][b[m]]
                if dd == dim:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1
                k = g
                for j in range(g, gg):
                    if count[gap[j]] == dim - dd:
                        gap[k] = gap[j]
                        k += 1
                    count[gap[j]] = 0
                g = k
        while l > 0 and g == gapter[l - 1]:
            l -= 1
            if l > 0:
                b[a[l]] = b[l]
                a[b[l]] = a[l]
        if l > 0:
            g -= 1
            a[l] = gap[g]
            b[l] = b[a[l]]
            b[a[l]] = l
            a[b[l]] = l
            gapter[l] = g
            l += 1

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    