# Notes and observations

## Key concepts

- A "leaf" is a unit square in the map
- A "gap" is a potential position where a new leaf can be folded
- Connections track how leaves can connect above/below each other
- The algorithm builds foldings incrementally by placing one leaf at a time
- Backtracking explores all valid combinations
- Leaves and dimensions are enumerated starting from 1, not 0; hence, leaf1ndex not leafIndex

## Algorithm flow

For each leaf:

- Find valid gaps in each dimension
- Place leaf in valid position
  - Try to find another leaf to put in the adjacent position
  - Repeat until the map is completely folded
- Backtrack when no valid positions remain

## Identifiers

Assume `listDimensions`:

1. is in an implemented OEIS sequence, and
2. is <= the first unknown in the OEIS sequence.

| Type     | max(x)      | Hunter Hogan            | alternative   | Lunnan | Irvine |
| -------- | ----------- | ----------------------- | ------------- | ------ | ------ |
| 3D array | n           | connectionGraph         |               | D      | d      |
| integer  | n           | computationalDivisions  | taskDivisions |        | mod    |
| integer  | n-1         | computationalIndex      | taskIndex     |        | res    |
| 2D array | max(p)      | coordinateSystem        |               | C      | c      |
| 1D array | n           | cumulativeProduct       |               | P      | bigP   |
| 1D array | d           | countDimensionsGapped   |               | count  | count  |
| integer  | d           | indexDimension          |               | i      | i      |
| integer  | d           | dimensionsUnconstrained |               | dd     | dd     |
| integer  | 8*          | dimensionsTotal         |               | d      | dim    |
| integer  |             | distance                |               | delta  | delta  |
| integer  | ~10^17      | foldingsTotal           | f             | G      | mCount |
| integer  |             | gap1ndex                |               | g      | g      |
| integer  |             | gap1ndexCeiling         |               | gg     | gg     |
| 1D array |             | gapRangeStart           |               | gapter | gapter |
| 1D array | n-1         | gapsWhere               |               | gap    | gap    |
| integer  | gg-1        | indexMiniGap            |               | j      | j      |
| integer  | n           | indexLeaf               |               | m      | m      |
| integer  | n+1         | leaf1ndex               |               | l      | l      |
| 1D array | n+1         | leafAbove               |               | A      | a      |
| 1D array | n+1         | leafBelow               |               | B      | b      |
| integer  | n           | leafConnectee           |               | m      | m      |
| integer  | n           | leafIndex               |               | m      | m      |
| integer  | 256*        | leavesTotal             |               | n      |        |
| 1D array | 19 (2x19)   | listDimensions          | mapShape      | p      | p      |
| 1D array | (container) | my                      |               |        |        |
| 1D array | (container) | the                     | static        |        |        |
| 2D array | (container) | track                   | s             |        |        |

*2x2x2x2x2x2x2x2 (2x2... 8-dimensional)

## "Known options" for `@numba.jit` decorator

| **Option**              | **Description**                                     | **Why**               | **Size**        | **But**                  |
| ----------------------- | --------------------------------------------------- | --------------------- | --------------- | ------------------------ |
| `_dbg_extend_lifetimes` | Debug option to extend object lifetimes             | Debugging             |                 |                          |
| `_dbg_optnone`          | Disable optimization for debugging                  | Debugging             |                 |                          |
| `debug`                 | Enable debug mode with additional checks            | Debugging             |                 |                          |
| `no_rewrites`           | Disable AST rewrites optimization                   | Debugging             |                 |                          |
| `boundscheck`           | Enable array bounds checking (slows execution)      | Error checking        | Larger          | Slower                   |
| `error_model`           | Divide by zero: kill or chill?                      | Error checking        | ?               |                          |
| `_nrt`                  | Enable No Runtime type checking                     | Startup speed         | Smaller         | No type protection       |
| `cache`                 | Caches compiled functions to disk                   | Faster reuse          |                 | Rare: need recompile     |
| `fastmath`              | Reduce float potential precision                    | Float speed           | Smaller         | Discriminatory, untested |
| `forceinline`           | Force function inlining                             | Reduce function calls | Likely larger   |                          |
| `forceobj`              | Force object mode compilation                       | Inclusiveness         | Larger          | Slower execution         |
| `inline`                | Algorithmically choose inlining                     | Speed                 | Slightly larger |                          |
| `looplift`              | Enable loop lifting optimization                    | Speed (if applicable) | Larger          | Exclusionary             |
| `no_cfunc_wrapper`      | Disable C function wrapper generation               | Size                  | Smaller         | Exclusionary             |
| `no_cpython_wrapper`    | Disable Python C-API wrapper generation             | Size                  | Smallest        | Exclusionary             |
| `nogil`                 | Releases Python GIL: parallel processes and threads | Concurrency           | Smaller         | No thread protection     |
| `nopython`              | Force pure machine code compilation                 | Speed                 |                 | Exclusionary             |
| `parallel`              | Enable automatic parallel execution                 | Concurrency           |                 | Rare: errors             |

`target`

## Miscellany

- All taskIndices can start from the states:
  - `gap1ndex > 0`
  - `not leaf1ndex != leavesTotal and leafConnectee % leavesTotal == leavesTotal - 1`
- 2 X n strip of stamps: "a(n), called G(n,2), is known to be divisible by 4n for n >= 2. - [Fred Lunnon](https://oeis.org/A001415), Dec 08 2013"
- The total number of folds is divisible by the total number of leaves.
- `for iteratee in incrementalRange`statements:
  - Disfavored because of dynamic memory allocation
  - Can easily be "reduced" to `while iteratee < firstExcludedValue`:
    - Once each runtime: Initialize permanently allocated memory for identifier `iteratee: type = 0`
    - Once each loop: Initialize value `iteratee = firstIncludedValue`
    - `while` loop:
      1. `while iteratee < firstExcludedValue` (or if you prefer, `while iteratee <= lastIncludedValue`)
      2. do pytastic pyStuffPy
      3. `iteratee = iteratee + step`
  - Hence, `while` has replaced `for`
  - Interestingly, the most deeply nested `while` loop, "while the connection-leaf is not equal to the active-leaf" (or `while m != l` or `while leafConnectee != leaf1ndex`), is actually a fancy `for` loop.
    - In the [original programming language](foldings.AA), the relationship was explicit:
      - for m := D[i,l,l], D[i,l,B[m]]
        - while m ≠ l do
    - In Python, we can see the "reduced" `for` loop:

    ```python
    # Initialize iteratee `m`
    m = D[i, l, l]
    while m != l:
        # do pytastic pyStuffPy
        m = D[i, l, B[m]]
    ```

    - With my identifiers:

    ```python
    # Initialize iteratee `leafConnectee`
    leafConnectee = connectionGraph[dimension1ndex, leaf1ndex, leaf1ndex]
    while leafConnectee != leaf1ndex:
        # do pytastic pyStuffPy
        leafConnectee = connectionGraph[dimension1ndex, leaf1ndex, leafBelow[leafConnectee]]
    ```

    - The `+ step` part, however, is well disguised, but compare the iteratee-initialization statement with the statement that changes the value of the iteratee:
      - `connectee = graph[d, leaf, leaf]`
      - `connectee = graph[d, leaf, leafBelow[connectee]]`
    - Or more abstracted:
      - `iteratee = value[first, included, included]`
      - `iteratee = value[first, included, different]`
    - `step` = the difference of the two statements
      - > `value[first, included, different]`
      - `- value[first, included, included]`
        - so `step = different - included`
        - or `step = leafBelow[connectee] - leaf`
        - or `step = leafBelow[leafConnectee] - leafConnectee`
        - or `step = B[m] - l`
- `countFolds` is the point of the package. Two things should be very stable
  1. the name of the function and
  2. the first parameter will accept a `list` of integers representing the dimensions of a map.
