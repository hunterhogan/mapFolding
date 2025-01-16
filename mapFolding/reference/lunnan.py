"""
An unnecessarily literal translation of the original Atlas Autocode code by W. F. Lunnon to Python.
W. F. Lunnon, Multi-dimensional map-folding, The Computer Journal, Volume 14, Issue 1, 1971, Pages 75-80, https://doi.org/10.1093/comjnl/14.1.75
"""# NOTE not functional yet
def foldings(p, job=None):
    """An unnecessarily literal translation of the original Atlas Autocode code."""
    p = list(p)
    p.append(None)  # NOTE mimics Atlas `array` type
    p.insert(0, None)  # NOTE mimics Atlas `array` type

    if job is None:
        global G
        G = 0
        def job(A, B):
            global G
            G = G + 1
        return foldings(p, job)
    # perform job (A, B) on each folding of a p[1] x ... x p[d] map,
    # where A and B are the above and below vectors. p[d + 1] < 0 terminates p;

    d: int
    n: int
    j: int
    i: int
    m: int
    l: int
    g: int
    gg: int
    dd: int

    n = 1
    i, d = 0, 0

    while (i := i + 1) and p[i] is not None:
        d = i
        n = n * p[i]

    # d dimensions and n leaves;

    # A: list[int] = [None] * (n + 1) # type: ignore
    # B: list[int] = [None] * (n + 1) # type: ignore
    # count: list[int] = [None] * (n + 1) # type: ignore
    # gapter: list[int] = [None] * (n + 1) # type: ignore
    # gap: list[int] = [None] * (n * n + 1) # type: ignore
    A: list[int] = [0] * (n + 1) # type: ignore
    B: list[int] = [0] * (n + 1) # type: ignore
    count: list[int] = [0] * (n + 1) # type: ignore
    gapter: list[int] = [0] * (n + 1) # type: ignore
    gap: list[int] = [0] * (n * n + 1) # type: ignore

    # B[m] is the leaf below leaf m in the current folding,
    # A[m] the leaf above. count[m] is the no. of sections in which
    # there is a gap for the new leaf l below leaf m,
    # gap[gapter[l - 1] + j] is the j-th (possible or actual) gap for leaf l,
    # and later gap[gapter[l]] is the gap where leaf l is currently inserted;

    P: list[int] = [0] * (d + 1) # type: ignore
    C: list[list[int]] = [[0] * (n + 1) for dimension1 in range(d + 1)] # type: ignore
    # D: list[list[list[int]]] = [[[None] * (n + 1) for dimension2 in range(n + 1)] for dimension1 in range(d + 1)] # type: ignore
    D: list[list[list[int]]] = [[[0] * (n + 1) for dimension2 in range(n + 1)] for dimension1 in range(d + 1)]

    P[0] = 1
    for i in range(1, d + 1):
        P[i] = P[i - 1] * p[i]

    for i in range(1, d + 1):
        for m in range(1, n + 1):
            C[i][m] = ((m - 1) // P[i - 1]) - ((m - 1) // P[i]) * p[i] + 1

    for i in range(1, d + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                D[i][l][m] = (0 if m == 0
                                else
                                        ((m if C[i][m] == 1
                                            else m - P[i - 1])
                                    if C[i][l] - C[i][m] == (C[i][l] - C[i][m]) // 2 * 2
                                    else
                                        (m if C[i][m] == p[i] or m + P[i - 1] > l
                                            else m + P[i - 1])))
    # P[i] = p[1] x ... x p[i], C[i][m] = i-th co-ordinate of leaf m,
    # D[i][l][m] = leaf connected to m in section i when inserting l;

    for m in range(n + 1):
        count[m] = 0

    A[0], B[0], g, l = 0, 0, 0, 0

    state = 'entry'
    while True:
        if state == 'entry':
            gapter[l] = g
            l = l + 1
            if l <= n:
                state = 'down'
                continue
            else:
                job(A, B)
                state = 'up'
                continue

        elif state == 'down':
            dd = 0
            gg = gapter[l - 1]
            g = gg
            for i in range(1, d + 1):
                if D[i][l][l] == l:
                    dd = dd + 1
                else:
                    m = D[i][l][l]
                    while m != l:
                        gap[gg] = m
                        if count[m] == 0:
                            gg = gg + 1
                        count[m] = count[m] + 1
                        m = D[i][l][B[m]]

            if dd == d:
                for m in range(l):
                    gap[gg] = m
                    gg = gg + 1

            for j in range(g, gg):
                gap[g] = gap[j]
                if count[gap[j]] == d - dd:
                    g = g + 1
                count[gap[j]] = 0
            state = 'along'
            continue

        elif state == 'along':
            if g == gapter[l - 1]:
                state = 'up'
                continue
            g = g - 1
            A[l] = gap[g]
            B[l] = B[A[l]]
            B[A[l]] = l
            A[B[l]] = l
            state = 'entry'
            continue

        elif state == 'up':
            l = l - 1
            B[A[l]] = B[l]
            A[B[l]] = A[l]
            if l > 0:
                state = 'along'
                continue
            else:
                break

    return G #if job.__closure__ else None
