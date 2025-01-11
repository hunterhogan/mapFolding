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
| integer  |             | activeGap1ndex          |               | g      | g      |
| integer  | n+1         | activeLeaf1ndex         |               | l      | l      |
| 3D array | n           | connectionGraph         |               | D      | d      |
| integer  | n           | computationalDivisions  | taskDivisions |        | mod    |
| integer  | n-1         | computationalIndex      | taskIndex     |        | res    |
| 2D array | max(p)      | coordinateSystem        |               | C      | c      |
| 1D array | n           | cumulativeProduct       |               | P      | bigP   |
| 1D array | d           | countDimensionsGapped   |               | count  | count  |
| integer  | d           | dimension1ndex          |               | i      | i      |
| integer  | d           | dimensionsUnconstrained |               | dd     | dd     |
| integer  | 8*          | dimensionsTotal         |               | d      | dim    |
| integer  |             | distance                |               | delta  | delta  |
| integer  | ~10^17      | foldingsTotal           | f             | G      | mCount |
| integer  |             | gap1ndexLowerBound      |               | gg     | gg     |
| 1D array |             | gapRangeStart           |               | gapter | gapter |
| integer  | gg-1        | indexMiniGap            |               | j      | j      |
| integer  | n           | index                   |               | m      | m      |
| 1D array | n+1         | leafAbove               |               | A      | a      |
| 1D array | n+1         | leafBelow               |               | B      | b      |
| integer  | n           | leaf1ndex               |               | m      | m      |
| integer  | n           | leaf1ndexConnectee      |               | m      | m      |
| integer  | 256*        | leavesTotal             |               | n      |        |
| 1D array | 19 (2x19)   | listDimensions          |               | p      | p      |
| 1D array | (container) | my                      |               |        |        |
| 1D array | > 2*n  <?   | potentialGaps           |               | gap    | gap    |
| 1D array | (container) | the                     | static        |        |        |
| 2D array | (container) | track                   | s             |        |        |

*2x2x2x2x2x2x2x2 (2x2... 8-dimensional)

## "Known options" for `@numba.jit` decorator

- '_dbg_extend_lifetimes',
- '_dbg_optnone',
- '_nrt',
- 'boundscheck', # Check for and report index errors
- 'debug',
- 'error_model',
- 'fastmath', # Disable CPU float precision
- 'forceinline',
- 'forceobj',
- 'inline',
- 'looplift',
- 'no_cfunc_wrapper',
- 'no_cpython_wrapper',
- 'no_rewrites',
- 'nogil',
- 'nopython', # Pure assembly
- 'parallel' # Enable automatic parallelization

## Miscellany

- All taskIndices can start from the states:
  - `activeGap1ndex > 0`
  - `not activeLeaf1ndex != leavesTotal and leaf1ndexConnectee % leavesTotal == leavesTotal - 1`
- 2 X n strip of stamps: "a(n), called G(n,2), is known to be divisible by 4n for n >= 2. - [Fred Lunnon](https://oeis.org/A001415), Dec 08 2013"
- The total number of folds is divisible by the total number of leaves.
