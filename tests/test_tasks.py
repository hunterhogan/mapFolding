from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from mapFolding import foldings, dimensionsFoldingsTotalLookup
import multiprocessing
import math

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

def concurrency_tests(concurrencyExecutor):
    foldingsTotal = 0
    dimensionsMap = [2, 7]
    expectedFoldingsTotal = dimensionsFoldingsTotalLookup[tuple(dimensionsMap)]
    with concurrencyExecutor() as concurrencyManager:
        listOfConcurrency = [
            concurrencyManager.submit(
                foldings, dimensionsMap, [leafNumber]
            )
            for leafNumber in range(1, math.prod(dimensionsMap) + 1)
        ]
        for index in listOfConcurrency:
            foldingsTotal += index.result()
    assert foldingsTotal == expectedFoldingsTotal

def test_concurrency_processes():
    concurrency_tests(ProcessPoolExecutor)

def test_concurrency_threads():
    concurrency_tests(ThreadPoolExecutor)

