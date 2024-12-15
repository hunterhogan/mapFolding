from mapFolding import oeisSequence_aOFn
from mapFolding.oeis import settingsOEISsequences
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import inspect
import numpy
import time
import pytest

def recordBenchmarks():
    """Decorator to benchmark a function."""
    def wrapper(functionTarget: Callable):
        def innerWrapper(*arguments, **keywordArguments):
            from . import _measureSpeed

            if not _measureSpeed:
                return functionTarget(*arguments, **keywordArguments)

            timeStart = time.perf_counter()
            result = functionTarget(*arguments, **keywordArguments)
            timeElapsed = time.perf_counter() - timeStart

            # Extract p and tasks from arguments
            p = tuple(sorted(arguments[2])) if len(arguments) >= 3 else None
            tasks = arguments[3] if len(arguments) >= 4 else None

            # Store benchmark data in single file
            pathBenchmarkFile = pathRecordedBenchmarks / "benchmarks.npy"
            benchmarkEntry = numpy.array([(timeElapsed, p, tasks if tasks is not None else 0)],
                                      dtype=[('time', 'f8'), ('dimensions', 'O'), ('tasks', 'i4')])
            
            if pathBenchmarkFile.exists():
                arrayExisting = numpy.load(str(pathBenchmarkFile), allow_pickle=True)
                arrayBenchmark = numpy.concatenate([arrayExisting, benchmarkEntry])
            else:
                arrayBenchmark = benchmarkEntry
            
            numpy.save(str(pathBenchmarkFile), arrayBenchmark)
            return result

        return innerWrapper
    return wrapper

"""
In general, benchmarking is passive: it records what happens and it doesn't initiate anything.
But, I can run this module and it will initiate calls to the algorithm by using the
defined benchmarkValues in the settingsOEISsequences dictionary. The recording process is still
the same, the only "active" role is calling the algorithm with the benchmark values.
`recordBenchmarks` decorates that which we want to measure.

Don't change "protected code", use it. The identifier `recordBenchmarks` is mandatory.
"""

# start protected code
benchmarkIterations = 32
pathRecordedBenchmarks = Path('mapFolding/benchmarks/marks')
pathRecordedBenchmarks.mkdir(parents=True, exist_ok=True)

@pytest.fixture(params=settingsOEISsequences.keys())
def oeisID(request):
    return request.param

def test_benchmarks(oeisID):
    for n in settingsOEISsequences[oeisID]['benchmarkValues']:
        oeisSequence_aOFn(oeisID, n)
# end protected code

