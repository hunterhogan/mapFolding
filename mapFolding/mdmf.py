"""5x5 186086600 161.66 seconds."""
from typing import List
from numba import njit

@njit
def calc_c(i: int, m: int, bigP: List[int], p: List[int]) -> int:
    """Calculate c[i][m] - i-th coordinate of leaf m"""
    return ((m - 1) // bigP[i-1]) - ((m - 1) // bigP[i]) * p[i-1] + 1

@njit
def calc_d(i: int, l: int, m: int, bigP: List[int], p: List[int]) -> int:
    """Calculate d[i][l][m] - leaf connected to m in section i when inserting l"""
    if m == 0:
        return 0
    delta = calc_c(i, l, bigP, p) - calc_c(i, m, bigP, p)
    if delta % 2 == 0:
        return m if calc_c(i, m, bigP, p) == 1 else m - bigP[i-1]
    return m if calc_c(i, m, bigP, p) == p[i-1] or m + bigP[i-1] > l else m + bigP[i-1]

@njit
def process_folding(a: List[int], b: List[int], n: int, normal: bool, count: List[int]) -> None:
    """Process a single folding if it meets criteria"""
    if not normal or b[0] == 1:
        count[0] += n

@njit
def fold_recursive(l: int, a: List[int], b: List[int], gap: List[int], 
                  count_arr: List[int], gapter: List[int], g: int,
                  n: int, dim: int, bigP: List[int], p: List[int],
                  normal: bool, count: List[int], part: int, parts: int) -> None:
    """Recursive folding implementation"""
    if l > n:
        process_folding(a, b, n, normal, count)
        return
        
    dd = 0
    gg = g = gapter[l-1]
    
    # Find gaps for leaf l
    for i in range(1, dim+1):
        m = calc_d(i, l, l, bigP, p)
        if m == l:
            dd += 1
        else:
            while m != l:
                if parts == 0 or l != parts or m % parts == part:
                    gap[gg] = m
                    if count_arr[m] == 0:
                        gg += 1
                    count_arr[m] += 1
                m = calc_d(i, l, b[m], bigP, p)
                
    # Process gaps
    if dd == dim:
        for m in range(l):
            gap[gg] = m
            gg += 1
            
    g = gapter[l-1]
    for j in range(g, gg):
        m = gap[j]
        gap[g] = m
        if count_arr[m] == dim - dd:
            g += 1
        count_arr[m] = 0
        
    # Try each gap
    old_g = g
    while g > gapter[l-1]:
        g -= 1
        a[l] = gap[g]
        b[l] = b[a[l]]
        b[a[l]] = l
        a[b[l]] = l
        gapter[l] = g
        fold_recursive(l+1, a, b, gap, count_arr, gapter, old_g,
                      n, dim, bigP, p, normal, count, part, parts)
        b[a[l]] = b[l]
        a[b[l]] = a[l]

@njit
def foldings(p: List[int], normal: bool = True, part: int = 0, parts: int = 0) -> int:
    """Calculate all possible foldings for dimensions specified in p."""
    n = 1
    for i in p:
        n *= i  # Total number of leaves
    dim = len(p)
    
    # Calculate bigP array
    bigP = [1]
    for i in range(dim):
        product = 1
        for j in range(i + 1):
            product *= p[j]
        bigP.append(product)
    
    # Initialize arrays
    a = [0] * (n+1)
    b = [0] * (n+1)
    count_arr = [0] * (n+1)
    gapter = [0] * (n+1)
    gap = [0] * (n*n+1)
    count = [0]
    
    # Start folding
    fold_recursive(1, a, b, gap, count_arr, gapter, 0,
                  n, dim, bigP, p, normal, count, part, parts)
    
    return count[0]

@njit
def count_foldings(p: List[int], normal: bool = True) -> int:
    """Count number of foldings for given dimensions"""
    return foldings(p, normal)
