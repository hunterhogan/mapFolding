import pathlib
import time
from typing import Callable

import numpy

pathRecordedBenchmarks = pathlib.Path('mapFolding/benchmarks/marks')
pathRecordedBenchmarks.mkdir(parents=True, exist_ok=True)
pathFilenameRecordedBenchmarks = pathRecordedBenchmarks / "benchmarks.npy"

def recordBenchmarks():
    """Decorator to benchmark a function."""
    def AzeemTheWrapper(functionTarget: Callable):
        def djZeph(*arguments, **keywordArguments):
            timeStart = time.perf_counter_ns()
            result = functionTarget(*arguments, **keywordArguments)
            timeElapsed = (time.perf_counter_ns() - timeStart) / 1e9

            # Extract listDimensions from arguments
            listDimensions = tuple(arguments[0])

            # Store benchmark data in single file
            benchmarkEntry = numpy.array([(timeElapsed, listDimensions)], dtype=[('time', 'f8'), ('dimensions', 'O')])
            
            if pathFilenameRecordedBenchmarks.exists():
                arrayExisting = numpy.load(str(pathFilenameRecordedBenchmarks), allow_pickle=True)
                arrayBenchmark = numpy.concatenate([arrayExisting, benchmarkEntry])
            else:
                arrayBenchmark = benchmarkEntry
            
            numpy.save(str(pathFilenameRecordedBenchmarks), arrayBenchmark)
            return result

        return djZeph
    return AzeemTheWrapper

def runBenchmarks(benchmarkIterations: int = 30) -> None:
    """Run benchmark iterations.
    
    Parameters:
        benchmarkIterations (30): Number of benchmark iterations to run
    """
    import itertools

    # TODO warmUp (False): Whether to perform one warm-up iteration
    from tqdm.auto import tqdm

    from mapFolding import oeisSequence_aOFn
    from mapFolding.oeis import settingsOEISsequences
    listParametersOEIS = [(oeisIdentifier, dimensionValue) for oeisIdentifier, settings in settingsOEISsequences.items() for dimensionValue in settings['valuesBenchmark']]
    for (oeisIdentifier, dimensionValue), iterationIndex in tqdm(itertools.product(listParametersOEIS, range(benchmarkIterations)), total=len(listParametersOEIS) * benchmarkIterations):
        oeisSequence_aOFn(oeisIdentifier, dimensionValue)

if __name__ == '__main__':
    pathFilenameRecordedBenchmarks.unlink(missing_ok=True)
    runBenchmarks(30)
