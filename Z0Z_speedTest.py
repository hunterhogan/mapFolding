from tqdm.auto import tqdm
import statistics
import time
from mapFolding import settingsOEISsequences, dimensionsFoldingsTotalLookup, countMinimumParsePoints
from Z0Z_dataAnalysis import processBenchmarkResults

# Import modules with folding functions
import mapFolding.mdmf

testCallables = [
    mapFolding.mdmf.foldings,
]

setTuplesTestDimensions = {tuple(sorted(settings['dimensions'](n))) for settings in settingsOEISsequences.values() for n in settings['testValuesSpeed']}
listTestDimensions = [list(testDimension) for testDimension in setTuplesTestDimensions]
testRounds = 3

listBenchmarkResults = []
for testDimension in tqdm(listTestDimensions, leave=False):
    for callableFunction in tqdm(testCallables, leave=False):
        foldingsTotal = callableFunction(testDimension)
        validTotal = dimensionsFoldingsTotalLookup[tuple(testDimension)]
        assert foldingsTotal == validTotal, f"Incorrect foldingsTotal. {callableFunction.__module__} {testDimension} {foldingsTotal} vs {validTotal}"
        
        listExecutionTimes = []
        for round in tqdm(range(testRounds), leave=False):  
            startTime = time.perf_counter()
            callableFunction(testDimension)
            listExecutionTimes.append(time.perf_counter() - startTime)
            
        if len(listExecutionTimes) >= 3:
            listExecutionTimes = sorted(listExecutionTimes)[1:-1]  # Remove min and max
        meanTime = statistics.mean(listExecutionTimes) * 1000
        
        listBenchmarkResults.append({
            "Dimension": tuple(testDimension),
            "Callable": callableFunction.__module__,
            "Mean Time (ms)": meanTime,
        })

processBenchmarkResults(listBenchmarkResults)