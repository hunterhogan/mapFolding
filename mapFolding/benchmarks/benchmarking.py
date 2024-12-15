from mapFolding import oeisSequence_aOFn
from mapFolding.oeis import settingsOEISsequences
from pathlib import Path
from typing import Callable
import numpy
import time

pathRecordedBenchmarks = Path('mapFolding/benchmarks/marks')
pathRecordedBenchmarks.mkdir(parents=True, exist_ok=True)

def recordBenchmarks():
    """Decorator to benchmark a function."""
    def wrapper(functionTarget: Callable):
        def innerWrapper(*arguments, **keywordArguments):
            timeStart = time.perf_counter()
            result = functionTarget(*arguments, **keywordArguments)
            timeElapsed = time.perf_counter() - timeStart

            # Extract p and tasks from arguments
            p = tuple(sorted(arguments[-2])) if len(arguments) >= 3 else None
            tasks = arguments[-1] if len(arguments) >= 4 else None

            # Store benchmark data in single file
            pathFilenameRecordedBenchmarks = pathRecordedBenchmarks / "benchmarks.npy"
            benchmarkEntry = numpy.array([(timeElapsed, p, tasks if tasks is not None else 0)],
                                      dtype=[('time', 'f8'), ('dimensions', 'O'), ('tasks', 'i4')])
            
            if pathFilenameRecordedBenchmarks.exists():
                arrayExisting = numpy.load(str(pathFilenameRecordedBenchmarks), allow_pickle=True)
                arrayBenchmark = numpy.concatenate([arrayExisting, benchmarkEntry])
            else:
                arrayBenchmark = benchmarkEntry
            
            numpy.save(str(pathFilenameRecordedBenchmarks), arrayBenchmark)
            return result

        return innerWrapper
    return wrapper

def runBenchmarks(benchmarkIterations: int = 30):
    """Run benchmark iterations with optional warm-up.
    
    Parameters
        benchmarkIterations (30): Number of benchmark iterations to run
        warmUp (False): Whether to perform one warm-up iteration
    """
    from tqdm.auto import tqdm

    listParameters = []
    for oeisID, settings in settingsOEISsequences.items():
        for n in settings['benchmarkValues']:
            listParameters.append((oeisID, n))

    for parameters in tqdm(listParameters):
        for iterationIndex in tqdm(range(benchmarkIterations), leave=False):
            oeisSequence_aOFn(*parameters)

if __name__ == '__main__':
    runBenchmarks(1)
