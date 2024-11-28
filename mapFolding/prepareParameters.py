from typing import Union
import multiprocessing

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

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
