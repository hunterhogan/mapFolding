from numba import njit

@njit
def foldings(p, flag=True, res=0, mod=0):
    """
    Calculate number of ways to fold a map with given dimensions.
    Parameters:
        p: array of dimensions [n, m] for nxm map or [n,n,n...] for n-dimensional
        flag: when True, only count "normal" foldings
        res/mod: specify which subset of foldings to compute (for parallel processing)
    Returns:
        total_count: Total number of valid foldings
    """
    # Calculate total number of leaves
    n = 1
    for dim in p:
        n *= dim

    # Initialize arrays needed for tracking the folding state
    a = [0] * (n + 1)  # leaf above m
    b = [0] * (n + 1)  # leaf below m
    count = [0] * (n + 1)  # Counts sections with gaps for new leaf
    gapter = [0] * (n + 1)  # Indices/pointers for each stack/level per leaf
    gap = [0] * (n * n + 1)  # All possible gaps for each leaf

    dim = len(p)  # number of dimensions

    # Calculate dimensional products and coordinates
    big_p = [1] * (dim + 1)
    for i in range(1, dim + 1):
        big_p[i] = big_p[i-1] * p[i-1]

    # Calculate coordinates in each dimension
    # c[i][m] holds the i-th coordinate of leaf m
    c = [[0] * (n + 1) for _ in range(dim + 1)]
    for i in range(1, dim + 1):
        for m in range(1, n + 1):
            c[i][m] = (m - 1) // big_p[i-1] - ((m - 1) // big_p[i]) * p[i-1] + 1

    # Calculate connections in each dimension
    # d[i][l][m] computes the leaf connected to m in section i when inserting l
    d = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(dim + 1)]
    for i in range(1, dim + 1):
        for l in range(1, n + 1):
            for m in range(1, l + 1):
                delta = c[i][l] - c[i][m]
                if delta % 2 == 0:
                    # If delta is even
                    d[i][l][m] = m if c[i][m] == 1 else m - big_p[i-1]
                else:
                    # If delta is odd
                    d[i][l][m] = m if (c[i][m] == p[i-1] or m + big_p[i-1] > l) else m + big_p[i-1]

    # Initialize variables for backtracking
    total_count = 0  # Total number of foldings
    g = 0            # Gap index
    l = 1            # Current leaf

    # Main folding loop using a stack-based approach
    while l > 0:
        if (not flag) or l <= 1 or b[0] == 1:
            if l > n:
                total_count += n
            else:
                dd = 0  # Number of sections where leaf l is unconstrained
                gg = gapter[l-1]  # track possible gaps
                g = gg

                # Find potential gaps for leaf l in each dimension
                for i in range(1, dim + 1):
                    if d[i][l][l] == l:
                        dd += 1
                    else:
                        m = d[i][l][l]
                        while m != l:
                            if mod == 0 or l != mod or m % mod == res:
                                gap[gg] = m
                                if count[m] == 0:
                                    gg += 1
                                count[m] += 1
                            m = d[i][l][b[m]]
                # If leaf l is unconstrained in all sections, it can be inserted anywhere
                # Handle unconstrained case and filter common gaps
                if dd == dim:
                    for m in range(l):
                        gap[gg] = m
                        gg += 1

                j = g
                while j < gg:
                    gap[g] = gap[j]
                    if count[gap[j]] == dim - dd:
                        g += 1
                    count[gap[j]] = 0
                    j += 1

        # Backtrack if no more gaps
        while l > 0 and g == gapter[l-1]:
            l -= 1
            b[a[l]] = b[l]
            a[b[l]] = a[l]

        # Insert leaf and advance
        if l > 0:
            g -= 1
            a[l] = gap[g]
            b[l] = b[a[l]]
            b[a[l]] = l
            a[b[l]] = l
            gapter[l] = g
            l += 1

    return total_count
