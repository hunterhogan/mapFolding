import jax
from typing import List
import jaxtyping

def foldings(listDimensions: List[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    from mapFolding.beDRY import validateParametersFoldings
    listDimensions, computationDivisions, computationIndex, n, connectionGraph = validateParametersFoldings(listDimensions, computationDivisions, computationIndex)
    
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

    dPlus1 = jax.numpy.add(d, 1)

    # How to build a connectionGraph: ("Cartesian Product Decomposition" or "Dimensional Product Mapping", allegedly)
    # Step 1: find the cumulative product of the map dimensions
    P = jax.numpy.cumprod(p, dtype=jax.numpy.int32)

    # Step 2: for each dimension, create a coordinate system
    # C[i][m] holds the i-th coordinate of leaf m
    C = jax.numpy.zeros((dPlus1, n + 1), dtype=jax.numpy.int32)
    for i in range(1, dPlus1):
        for m in range(1, n + 1):
            C = C.at[i, m].set(((m - 1) // P[i - 1]) % p[i - 1] + 1)
    """
    Key data structures
        - leafConnectionGraph[D][L][M]: How leaf L connects to leaf M in dimension D
        - track[count][L]: Number of dimensions with valid gaps at leaf L
        - track[gapter][L]: Index ranges of gaps available for leaf L
        - gap[]: List of all potential gap positions
    """

    # Step 3: create a huge empty leafConnectionGraph
    D = jax.numpy.zeros((dPlus1, n + 1, n + 1), dtype=jax.numpy.int32)
    # D[i][l][m] computes the leaf connected to m in dimension i when inserting l
    # Step for... for... for...: fill the leafConnectionGraph
    for i in range(1, dPlus1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = C[i, l] - C[i, m]
                if delta % 2 == 0:
                    # If delta is even
                    if C[i, m] == 1:
                        D = D.at[i, l, m].set(m)
                    else:
                        D = D.at[i, l, m].set(m - P[i - 1])
                else:
                    # If delta is odd
                    if C[i, m] == p[i - 1] or m + P[i - 1] > l:
                        D = D.at[i, l, m].set(m)
                    else:
                        D = D.at[i, l, m].set(m + P[i - 1])

    from mapFolding.pid import spoon
    return spoon(taskDivisions, arrayIndicesComputation, n, d, D)
