from mapFolding import computeSeries, computeSeriesConcurrently, countMinimumParsePoints, computeDistributedTask, \
    getDimensions, sumDistributedTasks
import time
import pathlib
def main():
    normalFoldings = True
    series = 'n'  # '2', '3', '2 X 2', or 'n'
    X_n = 4 # A non-negative integer

    timeStart = time.time()
    foldingsTotal = computeSeriesConcurrently(series, X_n)
    print(f"Dimensions: {series} X {X_n} = {foldingsTotal}. {time.time() - timeStart:.2f} seconds, concurrency.")

    timeStart = time.time()
    foldingsTotal = computeSeries(series, X_n)
    print(f"Dimensions: {series} X {X_n} = {foldingsTotal}. {time.time() - timeStart:.2f} seconds.")

def doTask():
    pathTasks = pathlib.Path("C:/apps/mapFolding/n/5/31/True")
    print(time.strftime('%H:%M:%S', time.localtime(timeStart := time.time())))
    tasksRemaining = computeDistributedTask(pathTasks, 3)
    timeStop = time.time()
    for pathFilename in pathTasks.iterdir():
        if pathFilename.stat().st_birthtime > timeStart:
            print(pathFilename, time.strftime('%H:%M:%S', time.localtime(pathFilename.stat().st_birthtime)))
    print(time.strftime('%H:%M:%S', time.localtime(timeStop)))
    print(f"{timeStop - timeStart:.0f} seconds.")
    print(f"Tasks remaining: {tasksRemaining}.")

def countEm():
    pp = pathlib.Path("G:/My Drive/dataHunter/mapFolding/n/5/31/True")
    print(sumDistributedTasks(pp))

def dd():
    series = 'n'
    X_n = 7
    dimensions = getDimensions(series, X_n)
    print(f"Dimensions: {series} X {X_n} = {dimensions}.")
    computationDivisions = countMinimumParsePoints(dimensions)
    print(f"Dimensions: {series} X {X_n} = {computationDivisions}.")
if __name__ == "__main__": 
    # dd()
    # countEm()
    main()

"""Results when mapFolding used the class MapFolding and the method computeSeriesConcurrently:
(mapFolding) C:/apps/mapFolding>Z0Z_getMapFoldings.py
Dimensions: n X 4 = 186086600. 32.57 seconds.

(mapFolding) C:/apps/mapFolding>Z0Z_getMapFoldings.py
Dimensions: n X 5 = 123912532224. 21040.25 seconds.
"""

"""Results 2024 November 27 with functional paradigm:
(mapFolding) C:/apps/mapFolding>Z0Z_getMapFoldings.py
Dimensions: n X 2 = 1368. 0.12 seconds.
Dimensions: n X 2 = 1368. 0.18 seconds, concurrency.

(mapFolding) C:/apps/mapFolding>Z0Z_getMapFoldings.py
Dimensions: n X 3 = 300608. 0.42 seconds.
Dimensions: n X 3 = 300608. 0.37 seconds, concurrency.

(mapFolding) C:/apps/mapFolding>Z0Z_getMapFoldings.py
Dimensions: n X 4 = 186086600. 32.53 seconds, concurrency.
Dimensions: n X 4 = 186086600. 141.23 seconds.
Dimensions: n X 4 = 186086600. 33.04 seconds, concurrency.
"""

"""Results 2024 November 28
Dimensions: n X 4 = 186086600. 19.16 seconds, concurrency.
Dimensions: n X 4 = 186086600. 72.32 seconds.
"""

"""Results 2024 November 29
from numba import njit
Dimensions: n X 4 = 186086600. 6.54 seconds, concurrency.
Dimensions: n X 4 = 186086600. 6.37 seconds.
"""