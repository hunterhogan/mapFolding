import multiprocessing

class MapFolding:
    """
    The algorithm enumerates all possible ways to fold a multi-dimensional map with given dimensions.

    Implements algorithm from 
    W. F. Lunnon, Multi-dimensional map-folding, "The Computer Journal", Volume 14, Issue 1, 1971, Pages 75-80, 
    https://doi.org/10.1093/comjnl/14.1.75 (see also "./citations/Lunnon.bibtex") 
    but directly based on Sean Irvine's Java port of Fred Lunnon's C version.

    See https://github.com/archmageirvine/joeis/blob/80e3e844b11f149704acbab520bc3a3a25ac34ff/src/irvine/oeis/a001/A001415.java
    """
    def __init__(self) -> None:
        self.count = 0

    def process(self, a, b, n: int) -> None:
        """Process each valid folding by incrementing the counter."""
        self.count += n

    def foldings(self, p: list[int], normalFoldings: bool = True, computationIndex: int = 0, computationDivisions: int = 0) -> None:
        """
        Perform enumeration of map foldings.
        
        Parameters:
            p: dimensions of the counting array, which may or may not be identical to 
                the dimensions of the map, see `getDimensions()`
            normalFoldings: when True, enumerate only normal foldings
            computationIndex: which part to compute (0 to computationDivisions-1)
            computationDivisions: split computation into this many parts. 
                 computationDivisions=0 means compute everything in one go
                 computationDivisions>1 means only compute part 'computationIndex' of 'computationDivisions' parts
                 Example: computationIndex=2, computationDivisions=5 computes the third fifth

        The algorithm uses the following key data structures:
            - b[m]: leaf below leaf m in current folding
            - a[m]: leaf above leaf m in current folding
            - count[m]: counts sections with gaps for new leaf l below leaf m
            - gap[gapter[l-1] + j]: holds j-th possible/actual gap for leaf l
        """
        # Calculate total number of leaves
        n = 1
        for pp in p:
            n *= pp

        # Initialize arrays
        a = [0] * (n + 1)  # leaf above
        b = [0] * (n + 1)  # leaf below
        count = [0] * (n + 1)
        gapter = [0] * (n + 1)
        gap = [0] * (n * n + 1)

        # Initialize dimension arrays
        dim = len(p)
        bigP = [1] * (dim + 1)
        c = [[0] * (n + 1) for _ in range(dim + 1)]
        d = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(dim + 1)]

        # Initialize bigP array with cumulative products
        for i in range(1, dim + 1):
            bigP[i] = bigP[i - 1] * p[i - 1]

        # Initialize c array - coordinate mapping
        for i in range(1, dim + 1):
            for m in range(1, n + 1):
                c[i][m] = (m - 1) // bigP[i - 1] - ((m - 1) // bigP[i]) * p[i - 1] + 1

        # Initialize d array - connection mapping
        for i in range(1, dim + 1):
            for l in range(1, n + 1):
                for m in range(1, l + 1):
                    delta = c[i][l] - c[i][m]
                    if (delta & 1) == 0:
                        d[i][l][m] = m if c[i][m] == 1 else m - bigP[i - 1]
                    else:
                        d[i][l][m] = m if c[i][m] == p[i - 1] or m + bigP[i - 1] > l else m + bigP[i - 1]

        g = 0  # Gap counter
        l = 1  # Current leaf

        # Main backtrack loop - implements Lunnon's state machine:
        # 1. Try to extend current folding by adding new leaf
        # 2. If no extension possible, backtrack
        # 3. Process completed foldings
        while l > 0:
            if not normalFoldings or l <= 1 or b[0] == 1:
                if l > n:
                    self.process(a, b, n)  # Found valid folding
                else:
                    dd = 0
                    gg = gapter[l - 1]
                    g = gg

                    # For each dimension, track gaps
                    for i in range(1, dim + 1):
                        if d[i][l][l] == l:
                            dd += 1
                        else:
                            m = d[i][l][l]
                            while m != l:
                                if computationDivisions == 0 or l != computationDivisions or m % computationDivisions == computationIndex:
                                    gap[gg] = m
                                    count[m] += 1  #Increment count here
                                    gg += 1 #Increment gg only if a new gap is added.
                                m = d[i][l][b[m]]

                    # Handle unconstrained case
                    if dd == dim:
                        for m in range(l):
                            gap[gg] = m
                            gg += 1

                    # Gap filtering and update
                    k = g
                    for j in range(g, gg):
                        if count[gap[j]] == dim - dd:
                            gap[k] = gap[j]
                            k += 1
                        count[gap[j]] = 0
                    g = k # Update g

            # Backtrack when no more gaps
            while l > 0 and g == gapter[l - 1]:
                l -= 1
                if l > 0:
                    b[a[l]] = b[l]
                    a[b[l]] = a[l]

            # Make next move if possible
            if l > 0:
                g -= 1
                a[l] = gap[g]
                b[l] = b[a[l]]
                b[a[l]] = l
                a[b[l]] = l
                gapter[l] = g
                l += 1

    def getDimensions(self, series: str, X_n: int) -> list[int]:  
        """
        Return the dimensions of the array to use to count the folds of a `series` `X_n` map.

        Parameters:
            series: the series of the map, e.g. '2', '3', '2 X 2', 'n'
            X_n: the number of dimensions, n, for the specified series

        Returns:
            dimensions: a list of integers that represent the size of each dimension in the counting array
        
        Explicitly implements dimensions for the following OEIS sequences
            A001415: 2 X n strip
            A001416: 3 X n strip
            A001417: 2 X 2 X ... X 2 (n-dimensional)
            A001418: n X n sheet
        """
        if isinstance(series, int):
            series = str(series)

        if series == '2':
            return [2, X_n]
        elif series == '3':
            return [3, X_n]
        elif series.lower() == '2 x 2':
            return [2] * X_n
        elif series == 'n':
            return [X_n + 1, X_n + 1]
        else:
            return [int(series), X_n]

    def computeSeries(self, series: str, X_n: int) -> int:
        """Computes the number of possible foldings for a given series and size.

        This method calculates the total number of valid ways to fold a map according to
        the specified series pattern and size.

        Parameters:
            series: the series of the map, e.g. '2', '3', '2 X 2', 'n'
            X_n: the number of dimensions, n, for the specified series

        Returns:
            foldingsTotal: The total number of folding combinations
        """
        self.count = 0
        if X_n == 0:
            return 1
        else:
            dimensions = self.getDimensions(series, X_n)
        self.foldings(dimensions, True, 0, 0)
        return self.count

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
