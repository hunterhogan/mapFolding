from typing import List, Callable
from functools import reduce
from operator import mul

def foldings(p: List[int], on_fold: Callable[[List[int], List[int], int], None], 
            normal: bool = True, part: int = 0, parts: int = 0) -> None:
    """Calculate all possible foldings for dimensions specified in p.
    
    Args:
        p: List of dimensions (each > 0)
        on_fold: Callback function receiving (above, below, n) for each valid folding
        normal: If True, only normal foldings are counted
        part/parts: For parallel processing
    """
    if not p:
        return
        
    n = reduce(mul, p, 1)  # Total number of leaves
    dim = len(p)
    
    # Calculate bigP, c, and d arrays
    bigP = [1] + [reduce(mul, p[:i+1], 1) for i in range(dim)]
    
    def calc_c(i: int, m: int) -> int:
        """Calculate c[i][m] - i-th coordinate of leaf m"""
        return ((m - 1) // bigP[i-1]) - ((m - 1) // bigP[i]) * p[i-1] + 1
        
    def calc_d(i: int, l: int, m: int) -> int:
        """Calculate d[i][l][m] - leaf connected to m in section i when inserting l"""
        if m == 0:
            return 0
        delta = calc_c(i, l) - calc_c(i, m)
        if delta % 2 == 0:
            return m if calc_c(i, m) == 1 else m - bigP[i-1]
        return m if calc_c(i, m) == p[i-1] or m + bigP[i-1] > l else m + bigP[i-1]

    def process_folding(a: List[int], b: List[int]) -> None:
        """Process a single folding if it meets criteria"""
        if not normal or b[0] == 1:
            on_fold(a, b, n)

    def fold_recursive(l: int, a: List[int], b: List[int], gap: List[int], 
                      count: List[int], gapter: List[int], g: int) -> None:
        """Recursive folding implementation"""
        if l > n:
            process_folding(a, b)
            return
            
        dd = 0
        gg = g = gapter[l-1]
        
        # Find gaps for leaf l
        for i in range(1, dim+1):
            m = calc_d(i, l, l)
            if m == l:
                dd += 1
            else:
                while m != l:
                    if parts == 0 or l != parts or m % parts == part:
                        gap[gg] = m
                        if count[m] == 0:
                            gg += 1
                        count[m] += 1
                    m = calc_d(i, l, b[m])
                    
        # Process gaps
        if dd == dim:
            for m in range(l):
                gap[gg] = m
                gg += 1
                
        g = gapter[l-1]
        for j in range(g, gg):
            m = gap[j]
            gap[g] = m
            if count[m] == dim - dd:
                g += 1
            count[m] = 0
            
        # Try each gap
        old_g = g
        while g > gapter[l-1]:
            g -= 1
            a[l] = gap[g]
            b[l] = b[a[l]]
            b[a[l]] = l
            a[b[l]] = l
            gapter[l] = g
            fold_recursive(l+1, a, b, gap, count, gapter, old_g)
            b[a[l]] = b[l]
            a[b[l]] = a[l]

    # Initialize arrays
    a = [0] * (n+1)
    b = [0] * (n+1)
    count = [0] * (n+1)
    gapter = [0] * (n+1)
    gap = [0] * (n*n+1)
    
    # Start folding
    fold_recursive(1, a, b, gap, count, gapter, 0)

def count_foldings(p: List[int], normal: bool = True) -> int:
    """Count number of foldings for given dimensions"""
    count = 0
    def counter(a, b, n):
        nonlocal count
        count += n
    foldings(p, counter, normal)
    return count

def is_symmetric(below: List[int], delta: int) -> bool:
    """Check if folding is symmetric (for A007822)"""
    n = len(below) - 1
    return all(below[(delta + k) % n] - ((delta + k) % n) == 
              below[(delta + n - 2 - k) % n] - ((delta + n - 2 - k) % n) 
              for k in range((n-1)//2))

def count_symmetric(n: int) -> int:
    """Count symmetric foldings (A007822)"""
    sym_count = 0
    def check_sym(_, b, __):
        nonlocal sym_count
        for k in range(len(b)):
            if is_symmetric(b, k):
                sym_count += 1
    foldings([n-1], check_sym)
    return (sym_count + 1) // 2
