import inspect
from dataclasses import dataclass
from pathlib import Path
import time
import json
from typing import Any, Callable, Dict, List, Optional, Union
import numpy

@dataclass
class ImplementationVariant:
    """Track a specific implementation variant."""
    name: str 
    function: Callable
    description: str
    parameters: Dict[str, Any]

@dataclass
class BenchmarkMeasurement:
    """Store results of a benchmark run."""
    name: str
    parameters: Dict[str, Any]
    countIterations: int
    listTimeNanoseconds: List[float]
    meanTimeNanoseconds: float
    standardDeviationNanoseconds: float
    timestampEpoch: float

def recordBenchmarks(function: Callable):
    """Decorator to benchmark a function."""
    # Get source info once during decoration
    sourceFunction = inspect.getsource(function)
    moduleFunction = getattr(inspect.getmodule(function), '__name__', '<unknown>')
    signatureFunction = inspect.signature(function)
    
    def wrapper(*arguments, **keywordArguments):
        from . import _flagEnableBenchmarks

        # Execute actual function
        result = function(*arguments, **keywordArguments)

        # Record benchmark if enabled
        if _flagEnableBenchmarks:
            def executeBenchmark():
                managerBenchmark = BenchmarkManager()
                managerBenchmark.managerRegistersImplementation(
                    f"benchmark_{function.__name__}",
                    function,
                    f"Benchmarking {moduleFunction}.{function.__name__}",
                    {
                        "module": moduleFunction,
                        "signature": str(signatureFunction),
                        "source": sourceFunction,
                    }
                )
                managerBenchmark.managerExecutesBenchmark(
                    f"benchmark_{function.__name__}",
                    [arguments],
                    countIterations=1  # Reduced iterations from 100 to 10
                )

            from threading import Thread
            threadBenchmark = Thread(target=executeBenchmark)
            threadBenchmark.start()

        return result
    
    return wrapper

class BenchmarkManager:
    """Manage benchmark implementations and results."""
    def __init__(self, pathResults: Optional[Path] = None):
        self.pathResults = pathResults or Path(__file__).parent / "results"
        self.pathResults.mkdir(parents=True, exist_ok=True)
        self.dictionaryImplementations: Dict[str, ImplementationVariant] = {}
        self.flagEnableBenchmarks = False  # Only record when explicitly enabled

    def managerRegistersImplementation(self, name: str, function: Callable, 
                              description: str, parameters: Dict[str, Any]) -> None:
        """Register a new implementation variant."""
        self.dictionaryImplementations[name] = ImplementationVariant(
            name, function, description, parameters)

    def managerExecutesBenchmark(self, name: str, listTestCases: List[Any], 
                     countIterations: int = 1) -> BenchmarkMeasurement:
        """Run benchmark for a specific implementation."""
        implementation = self.dictionaryImplementations[name]
        listTimeNanoseconds = []

        # Warmup run
        for testCase in listTestCases[:1]:
            implementation.function(*testCase)

        # Timed runs
        for iterator in range(countIterations):
            timeStart = time.perf_counter_ns()
            for testCase in listTestCases:
                implementation.function(*testCase)
            listTimeNanoseconds.append(
                (time.perf_counter_ns() - timeStart) / len(listTestCases))

        arrayTimes = numpy.array(listTimeNanoseconds)
        measurement = BenchmarkMeasurement(
            name=name,
            parameters=implementation.parameters,
            countIterations=countIterations,
            listTimeNanoseconds=listTimeNanoseconds,
            meanTimeNanoseconds=float(numpy.mean(arrayTimes)),
            standardDeviationNanoseconds=float(numpy.std(arrayTimes)),
            timestampEpoch=time.time()
        )

        self._managerSavesMeasurement(measurement)
        return measurement

    def _managerSavesMeasurement(self, measurement: BenchmarkMeasurement) -> None:
        """Save benchmark results to JSON file."""
        filenameResult = f"{measurement.name}_{int(measurement.timestampEpoch)}.json"
        with open(self.pathResults / filenameResult, 'w') as writeStream:
            json.dump(measurement.__dict__, writeStream, indent=2)
