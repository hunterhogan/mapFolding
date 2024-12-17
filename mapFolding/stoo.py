import jax

jax.config.update("jax_enable_x64", True)

def foldings(listDimensions: list[int], computationDivisions: int = 0, computationIndex: int = 0) -> int:
    from mapFolding.beDRY import validateParametersFoldings
    listDimensions, computationDivisions, computationIndex, n = validateParametersFoldings(listDimensions, computationDivisions, computationIndex)

    n = jax.numpy.array(n, dtype=jax.numpy.int64)
    d = jax.numpy.array(len(listDimensions), dtype=jax.numpy.int64)
    taskDivisions = jax.numpy.array(computationDivisions, dtype=jax.numpy.int64)
    del computationDivisions

    if taskDivisions < 2:
        taskDivisions = n
        del computationIndex
        arrayIndicesComputation = jax.numpy.arange(taskDivisions)
    else:
        arrayIndicesComputation = jax.numpy.array(computationIndex, dtype=jax.numpy.int64)
        del computationIndex

    p = jax.numpy.array(listDimensions, dtype=jax.numpy.int64)

    P = jax.numpy.ones(d + 1, dtype=jax.numpy.int64)
    for i in range(1, d + 1):
        P = P.at[i].set(P[i - 1] * p[i - 1])

    # C[i][m] holds the i-th coordinate of leaf m
    C = jax.numpy.zeros((d + 1, n + 1), dtype=jax.numpy.int64)
    for i in range(1, d + 1):
        for m in range(1, n + 1):
            C = C.at[i, m].set(((m - 1) // P[i - 1]) % p[i - 1] + 1)

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
                    if C[i, m] == p[i - 1] or m + P[i - 1] > l:
                        D = D.at[i, l, m].set(m)
                    else:
                        D = D.at[i, l, m].set(m + P[i - 1])

    from mapFolding.pid import spoon
    return spoon(taskDivisions, arrayIndicesComputation, n, d, D)
