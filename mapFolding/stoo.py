import jax
from typing import List
import jaxtyping

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    from mapFolding.beDRY import validateParametersFoldings
    listDimensions, computationDivisions, computationIndex, n, connectionGraph = validateParametersFoldings(listDimensions, computationDivisions, computationIndex)
    D = jax.numpy.array(connectionGraph, dtype=jax.numpy.int32)
    d = jax.numpy.int32(len(listDimensions))
    taskDivisions = jax.numpy.int32(computationDivisions)
    del computationDivisions

    p = jax.numpy.array(listDimensions, dtype=jax.numpy.int32)
    n = jax.numpy.prod(p, where=p > 0)

    if taskDivisions < 2:
        taskDivisions = n
        del computationIndex
        arrayIndicesComputation = jax.numpy.arange(taskDivisions, dtype=jax.numpy.int32)
    else:
        arrayIndicesComputation = jax.numpy.array(computationIndex, dtype=jax.numpy.int32)
        del computationIndex

    """
    Key data structures
        - leafConnectionGraph[D][L][M]: How leaf L connects to leaf M in dimension D
        - track[count][L]: Number of dimensions with valid gaps at leaf L
        - track[gapter][L]: Index ranges of gaps available for leaf L
        - gap[]: List of all potential gap positions
    """

    from mapFolding.pid import spoon
    return spoon(taskDivisions, arrayIndicesComputation, n, d, D)
