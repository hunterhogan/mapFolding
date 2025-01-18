"""An incompetent benchmarking module for mapFolding."""
from typing import Callable
import multiprocessing
import numpy
import pathlib
import time

pathRecordedBenchmarks = pathlib.Path('mapFolding/benchmarks/marks')
pathRecordedBenchmarks.mkdir(parents=True, exist_ok=True)
pathFilenameRecordedBenchmarks = pathRecordedBenchmarks / "benchmarks.npy"

def recordBenchmarks():
    """Decorator to benchmark a function."""
    def AzeemTheWrapper(functionTarget: Callable):
        def djZeph(*arguments, **keywordArguments):
            timeStart = time.perf_counter_ns()
            returnValueTarget = functionTarget(*arguments, **keywordArguments)
            timeElapsed = (time.perf_counter_ns() - timeStart) / 1e9

            # Extract mapShape from arguments
            mapShape = keywordArguments['mapShape']
            # mapShape = tuple(arguments)[2]
            # leavesTotal = tuple(arguments[3])[4]

            # Store benchmark data in single file
            benchmarkEntry = numpy.array([(timeElapsed, mapShape)], dtype=[('time', 'f8'), ('mapShape', 'O')])
            # benchmarkEntry = numpy.array([(timeElapsed, leavesTotal)], dtype=[('time', 'f8'), ('leaves', 'O')])

            if pathFilenameRecordedBenchmarks.exists():
                arrayExisting = numpy.load(str(pathFilenameRecordedBenchmarks), allow_pickle=True)
                arrayBenchmark = numpy.concatenate([arrayExisting, benchmarkEntry])
            else:
                arrayBenchmark = benchmarkEntry

            numpy.save(str(pathFilenameRecordedBenchmarks), arrayBenchmark)
            return returnValueTarget

        return djZeph
    return AzeemTheWrapper

def runBenchmarks(benchmarkIterations: int = 30) -> None:
    """Run benchmark iterations.

    Parameters:
        benchmarkIterations (30): Number of benchmark iterations to run
    """
    # TODO warmUp (False): Whether to perform one warm-up iteration

    import itertools
    from tqdm.auto import tqdm
    from mapFolding.oeis import settingsOEIS, oeisIDfor_n
    from concurrent.futures import ProcessPoolExecutor, as_completed
    max_workers = 6

    listParametersOEIS = [(oeisIdentifier, dimensionValue) for oeisIdentifier, settings in settingsOEIS.items() for dimensionValue in settings['valuesBenchmark']]
    # for (oeisIdentifier, dimensionValue), iterationIndex in tqdm(itertools.product(listParametersOEIS, range(benchmarkIterations)), total=len(listParametersOEIS) * benchmarkIterations):
    #     oeisIDfor_n(oeisIdentifier, dimensionValue)
    listCartesianProduct = list(itertools.product(listParametersOEIS, range(benchmarkIterations)))
    with ProcessPoolExecutor(max_workers) as concurrencyManager:
        listConcurrency = [concurrencyManager.submit(oeisIDfor_n, *parameters[0]) for parameters in listCartesianProduct]
        for _complete in tqdm(as_completed(listConcurrency), total=len(listCartesianProduct)):
            pass

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pathFilenameRecordedBenchmarks.unlink(missing_ok=True)
    runBenchmarks(30)
