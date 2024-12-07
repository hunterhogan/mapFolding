from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from mapFolding import foldings, dimensionsFoldingsTotalLookup
import multiprocessing
import math

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    foldingsTotal = 0
    dimensionsMap = [5, 5]
    knownTotal = dimensionsFoldingsTotalLookup[tuple(dimensionsMap)]
    computationalDivisions = math.prod(dimensionsMap)
    with ProcessPoolExecutor() as concurrencyManager:
        listOfConcurrency = [concurrencyManager.submit(foldings, dimensionsMap, [leafNumber]) for leafNumber in range(1, computationalDivisions+1)]

        for index in tqdm(as_completed(listOfConcurrency), total=computationalDivisions):
            foldingsTotal = foldingsTotal + index.result()

    print(f"{foldingsTotal=}", foldingsTotal==knownTotal)