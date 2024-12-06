from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from mapFolding import foldings, countMinimumParsePoints, dimensionsFoldingsTotalLookup
import multiprocessing
import math

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    foldingsTotal = 0
    dimensionsMap = [2, 15]
    knownTotal = dimensionsFoldingsTotalLookup[tuple(dimensionsMap)]
    computationalDivisions = max(countMinimumParsePoints(dimensionsMap), math.prod(dimensionsMap))
    with ProcessPoolExecutor() as concurrencyManager:
        listOfConcurrency = [concurrencyManager.submit(foldings, dimensionsMap, computationalDivisions, index) for index in range(computationalDivisions)]

        for index in tqdm(as_completed(listOfConcurrency), total=computationalDivisions):
            foldingsTotal = foldingsTotal + index.result()

    print(f"{foldingsTotal=}", foldingsTotal==knownTotal)