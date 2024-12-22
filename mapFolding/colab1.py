from mapFolding.benchmarks import recordBenchmarks
from mapFolding.colab3 import taskDivisions, taskIndex, leavesTotal, dimensionsTotal
from typing import List
import numpy
from numba import cuda

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    from mapFolding.beDRY import validateParametersFoldings
    listDimensions, computationDivisions, computationIndex, n, D = validateParametersFoldings(listDimensions, computationDivisions, computationIndex)

    d = len(listDimensions)  # Number of dimensions

    track = numpy.zeros((4, n + 1), dtype=numpy.int64)
    gap = numpy.zeros(n * n + 1, dtype=numpy.int64) # Stack of potential gaps
    static = numpy.zeros(4, dtype=numpy.int64)
    static[taskDivisions] = computationDivisions
    static[taskIndex] = computationIndex
    static[leavesTotal] = n
    static[dimensionsTotal] = d
    # Pass listDimensions and taskDivisions to _sherpa for benchmarking
    foldingsTotal = _sherpa(track, gap, static, D, listDimensions, computationDivisions)
    return foldingsTotal

@recordBenchmarks()
def _sherpa(track: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], gap: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], static: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], D: numpy.ndarray[numpy.int64, numpy.dtype[numpy.int64]], p: List[int], tasks: int) -> int:
    """Performance critical section that counts foldings."""
    from mapFolding.colab2 import countFoldings
    
    # Create array for results
    result = numpy.zeros(1, dtype=numpy.int64)
    
    # Copy arrays to device
    track_device = cuda.to_device(track)
    gap_device = cuda.to_device(gap)
    static_device = cuda.to_device(static)
    D_device = cuda.to_device(D)
    result_device = cuda.to_device(result)
    
    # Configure the blocks and threads
    threadsperblock = 256
    blockspergrid = 1
    
    # Launch kernel
    countFoldings[blockspergrid, threadsperblock](track_device, gap_device, static_device, D_device, result_device)
    
    # Copy result back
    result = result_device.copy_to_host()
    
    return int(result[0])
