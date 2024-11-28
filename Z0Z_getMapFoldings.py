from mapFolding import computeSeries, computeSeriesConcurrently
import time
def main():
    normalFoldings = True
    series = '3'  # '2', '3', '2 X 2', or 'n'
    X_n = 3 # A non-negative integer

    timeStart = time.time()
    foldingsTotal = computeSeriesConcurrently(series, X_n)
    print(f"Dimensions: {series} X {X_n} = {foldingsTotal}. {time.time() - timeStart:.2f} seconds, concurrency.")

    timeStart = time.time()
    foldingsTotal = computeSeries(series, X_n)
    print(f"Dimensions: {series} X {X_n} = {foldingsTotal}. {time.time() - timeStart:.2f} seconds.")

if __name__ == "__main__":
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