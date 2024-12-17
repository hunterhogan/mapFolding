from typing import List, Tuple
import jax
jax.config.update("jax_enable_x64", True)

def foldings(p: list[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    listDimensions = _validateListDimensions(p)
    del p
    from mapFolding import  getLeavesTotal
    n = getLeavesTotal(listDimensions)
    computationDivisions, computationIndex = _validateTaskDivisions(computationDivisions, computationIndex, n)
    if computationDivisions < 2:
        computationDivisions = n
        del computationIndex
        arrayIndicesComputation = jax.numpy.arange(computationDivisions)

    d = len(listDimensions)

    P = jax.numpy.ones(d + 1, dtype=jax.numpy.int64)
    for i in range(1, d + 1):
        P = P.at[i].set(P[i - 1] * listDimensions[i - 1])

    # C[i][m] holds the i-th coordinate of leaf m
    C = jax.numpy.zeros((d + 1, n + 1), dtype=jax.numpy.int64)
    for i in range(1, d + 1):
        for m in range(1, n + 1):
            C = C.at[i, m].set(((m - 1) // P[i - 1]) % listDimensions[i - 1] + 1)

    # D[i][l][m] computes the leaf connected to m in section i when inserting l
    D = jax.numpy.zeros((d + 1, n + 1, n + 1), dtype=jax.numpy.int64)
    for i in range(1, d + 1):
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
                    if C[i, m] == listDimensions[i - 1] or m + P[i - 1] > l:
                        D = D.at[i, l, m].set(m)
                    else:
                        D = D.at[i, l, m].set(m + P[i - 1])

    taskDivisions = jax.numpy.array(computationDivisions, dtype=jax.numpy.int64)
    n = jax.numpy.array(n, dtype=jax.numpy.int64)
    d = jax.numpy.array(d, dtype=jax.numpy.int64)
    from .pid import spoon
    return spoon(taskDivisions, arrayIndicesComputation, n, d, D)

def _validateTaskDivisions(computationDivisions: int, computationIndex: int, n: int) -> Tuple[int, int]:
    if computationDivisions > n:
        raise ValueError(f"computationDivisions, {computationDivisions}, must be less than or equal to the total number of leaves, {n}.")
    if computationDivisions > 1 and computationIndex >= computationDivisions:
        raise ValueError(f"computationIndex, {computationIndex}, must be less than computationDivisions, {computationDivisions}.")
    if computationDivisions < 0 or computationIndex < 0 or not isinstance(computationDivisions, int) or not isinstance(computationIndex, int):
        raise ValueError(f"computationDivisions, {computationDivisions}, and computationIndex, {computationIndex}, must be non-negative integers.")
    return computationDivisions, computationIndex

def _validateListDimensions(listDimensions: List[int]) -> List[int]:
    from mapFolding import parseListDimensions
    if listDimensions is None:
        raise ValueError(f"listDimensions is a required parameter.")
    listNonNegative = parseListDimensions(listDimensions, 'listDimensions')
    listPositive = [dimension for dimension in listNonNegative if dimension > 0]
    if len(listPositive) < 2:
        from typing import get_args
        from mapFolding.oeis import OEISsequenceID
        raise NotImplementedError(f"This function requires listDimensions, {listDimensions}, to have at least two dimensions greater than 0. Other functions in this package implement the sequences {get_args(OEISsequenceID)}. You may want to look at https://oeis.org/.")
    listDimensions = listPositive
    return listDimensions
