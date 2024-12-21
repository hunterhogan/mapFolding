import jax
from typing import List
import jaxtyping # TODO learn how to use this
import numpy as NUMERICALPYTHON
from mapFolding.piderIndices import taskDivisions, leavesTotal, dimensionsTotal

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0):
    from mapFolding.beDRY import validateParametersFoldings

    sherpa_the = NUMERICALPYTHON.zeros(3, dtype=NUMERICALPYTHON.int32)
    listDimensions, sherpa_the[int(taskDivisions)], computationIndex, sherpa_the[int(leavesTotal)], D = validateParametersFoldings(listDimensions, computationDivisions, computationIndex)

    sherpa_the[int(dimensionsTotal)] = len(listDimensions)
    
    connectionGraph = jax.numpy.array(D, dtype=jax.numpy.int32)

    del computationDivisions

    if sherpa_the[int(taskDivisions)] < 2:
        sherpa_the[int(taskDivisions)] = sherpa_the[int(leavesTotal)]
        arrayIndicesComputation = jax.numpy.arange(sherpa_the[int(taskDivisions)], dtype=jax.numpy.int32)
    else:
        arrayIndicesComputation = jax.numpy.array(computationIndex, dtype=jax.numpy.int32)
    del computationIndex

    the = jax.numpy.array(sherpa_the, dtype=jax.numpy.int32)

    track = jax.numpy.zeros((4, jax.numpy.add(the[leavesTotal], 1)), dtype=jax.numpy.int32)
    potentialGapsLength = the[leavesTotal] * the[leavesTotal] + 1
    potentialGaps = jax.numpy.zeros(potentialGapsLength, dtype=jax.numpy.int32)

    from mapFolding.pider import spoon
    return spoon(connectionGraph, the, track, potentialGaps, arrayIndicesComputation)
