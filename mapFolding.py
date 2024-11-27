from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union
import multiprocessing
import os
import pathlib
import random

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

    def computeSeries(self, series: str, X_n: int) -> int:
        """Computes the number of possible foldings for a given series and size.

        Parameters:
            series: the series of the map, e.g. '2', '3', '2 X 2', 'n'
            X_n: the number of dimensions, n, for the specified series

        Returns:
            foldingsTotal: The total number of folding combinations
        """
        return self.computeSeriesConcurrently(series, X_n, True, 1)

    def computeSeriesConcurrently(self, series: str, X_n: int, normalFoldings: bool = True, CPUlimit: Optional[Union[int, float, bool]] = None) -> int:
        """
        Parameters:
            CPUlimit (gluttonous resource usage): The maximum number of concurrent processes to use (default is None, which uses the number of CPUs).
        """
        # I want a better way to generalize "2 X n" and such.
        # TODO: make docstring available without duplicating writing the docstring
        if CPUlimit is None:
            CPUlimit = 0
        max_workers = self.getCPUlimit(CPUlimit)

        self.count = 0
        if X_n == 0:
            return 1
        else:
            dimensions = self.getDimensions(series, X_n)

        computationDivisions = self.countMinimumParsePoints(dimensions)

        with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
            listConcurrency = [concurrencyManager.submit(self.computeSeriesTask, dimensions, normalFoldings, computationIndex, computationDivisions)                               
                                  for computationIndex in range(computationDivisions)]
            listPortionValues = [concurrency.result() for concurrency in listConcurrency]

        self.count = sum(listPortionValues)

        return self.count

    def getCPUlimit(self, CPUlimit: Union[int, float, bool]) -> int:
        """
        Determines the maximum number of CPU workers to use based on the provided CPU limit.

        Parameters:
            CPUlimit: The limit for CPU usage.
                - If None, the maximum number of CPU workers is set to the total number of CPUs available.
                - If a boolean and True, the maximum number of CPU workers is set to 1.
                - If an integer:
                    - If greater than 0, the maximum number of CPU workers is set to the provided integer.
                    - If equal to 0, the maximum number of CPU workers remains unchanged.
                    - If less than 0, the maximum number of CPU workers is reduced by the magnitude of the integer.
                - If a float, the maximum number of CPU workers is set to the product of the provided float and the total number of CPUs available, with a minimum of 1.

        Returns:
            max_workers: The maximum number of CPU workers to use.
        """
        max_workers = multiprocessing.cpu_count()
        if CPUlimit is not None:
            if isinstance(CPUlimit, bool):
                if CPUlimit == True:
                    max_workers = 1
            elif isinstance(CPUlimit, int):
                if CPUlimit > 0:
                    max_workers = CPUlimit
                elif CPUlimit == 0:
                    pass
                elif CPUlimit < 0:
                    max_workers = max(multiprocessing.cpu_count() + CPUlimit, 1)
            elif isinstance(CPUlimit, float):
                max_workers = max(int(CPUlimit * multiprocessing.cpu_count()), 1)
        return max_workers

    def countMinimumParsePoints(self, dimensions: list[int]) -> int:
        """Calculates the minimum number of parse points needed for map folding.
        This method determines the minimum number of points where parallel computation
        can safely parse the folding calculation tree without overcounting. It analyzes
        the dimensions of the map to find positions where the folding process can be
        divided into concurrent subtasks.
        Parameters:
            dimensions: List of integers representing the dimensions of the map.
                Each integer represents the size of one dimension.
        Returns:
            COUNTreachesParsePoint: The minimum number of parse points available for safe concurrent computation
                of map foldings.
        """ 
        # Calculate total leaves
        leavesTotal = 1
        for dimensionSize in dimensions:
            leavesTotal *= dimensionSize
            
        """If the number of `computationDivisions` is more than the number
        of times we reach the parse point, the concurrent processes will
        overcount the foldings. If a subtree will cross the parse point at
        least once, it is safe to divide the computation once. So, the minimum
        number of subtrees that can be parsed out is the maximum number of
        parsings: `computerDivisions`. *

        If this is true, perhaps generalize the calculation to k-degrees, then pass k as a parameter.

        Or, if the entire problem can be reliably divided into the exact number of divisions, then
        the first step should be to devolve the problem into the divisions. Second, use simplier 
        logic to count the foldings in a division. Third, sum the results of the divisions. Therefore,
        it's nearly certain that we don't know how to reliably calculate the number of divisions.

        But, is it possible to: calculate the 1st degree of divisions. Then each time the worker arrives at the 
        parse point, the worker stops, divides their current tree into one more degree of divisions, and
        "records" the entry point to those divisions.
        """
            
        # For each leaf for each dimension of the map, do we reach the parse point at least once
        COUNTreachesParsePoint = sum(1 for potentialDivision in range(1, leavesTotal + 1) 
                                    if any(potentialDivision == 1 or potentialDivision - dimensionSize >= 1 
                                          for dimensionSize in dimensions))
        return COUNTreachesParsePoint

    def getDimensions(self, series: str, X_n: int) -> list[int]:  
        """
        Return the dimensions of the array used to count the folds of a `series` `X_n` map.
        
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

    def computeSeriesTask(self, dimensions: list[int], normalFoldings: bool, computationIndex: int, computationDivisions: int) -> int:
        self.count = 0
        self.foldings(dimensions, normalFoldings, computationIndex, computationDivisions)
        return self.count

    def computeDistributedTask(self, pathTasks: Union[str, os.PathLike[str]], CPUlimit: Optional[Union[int, float, bool]] = None) -> None:
        """
        Computes distributed tasks for map folding calculations.

        Parameters:
            pathTasks:
                Path containing task instructions encoded in format
                'series/X_n/normalFoldings/computationDivisions'
            CPUlimit (gluttonous resource usage): CPU core limit for parallel processing.
                See `getCPUlimit()` for interpretation details.
        Returns:
            None: Results are written directly to files in the `pathTasks` directory. Each completed computation is saved as a separate file named by its index
        """
        # TODO add unit tests for this method
        if CPUlimit is None:
            CPUlimit = 0
        max_workers = self.getCPUlimit(CPUlimit)

        series, X_n, normalFoldings, computationDivisions = pathlib.PurePosixPath(pathTasks).parts[-4:]

        dimensions = self.getDimensions(series, int(X_n))

        listComputationIndices = list(range(int(computationDivisions)))
        for indexCompleted in pathlib.Path(pathTasks).glob('*'):
            listComputationIndices.remove(int(indexCompleted.stem))
        random.shuffle(listComputationIndices)
        listIndicesTasks = listComputationIndices[0:max_workers]

        with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
            dictionaryConcurrency = {concurrencyManager.submit(
                self.computeSeriesTask, dimensions, bool(normalFoldings), computationIndex, int(computationDivisions))
                                     : computationIndex for computationIndex in listIndicesTasks}

            for claimTicket in as_completed(dictionaryConcurrency):
                indexCompleted = dictionaryConcurrency[claimTicket]
                pathlib.Path(pathTasks, str(indexCompleted)).write_text(str(claimTicket.result()))
        
    def process(self, a, b, n: int) -> None:
        """Process each valid folding by incrementing the counter."""
        # I haven't figured out why `a` and `b` are passed as arguments.
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
                                # The parse point
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

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

# * Or, I don't know what the hell I'm talking about.