# Notes and observations

A markdown file is an absurd way to track this information in 2025.

## Some concepts

- A "leaf" is a unit square in the map
- A "gap" is a potential position where a new leaf can be folded
- Connections track how leaves can connect above/below each other
- Leaves are enumerated starting from 1, not 0; hence, `leaf1ndex` not `leafIndex` (`1ndexLeaf` is not a valid Python identifier because it starts with a number)

## Algorithm flow

For each leaf:

- Find valid gaps in each dimension
- Place leaf in valid position
  - Try to find another leaf to put in the adjacent position
  - Repeat until the map is completely folded
- Backtrack when no valid positions remain

## Identifiers and their sizes

I would like to make a comprehensive and authoritative list of variables and formulas for their maximum bit-width.

## Replace `for` with `while`

- `for iteratee in incrementalRange` statements:
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

## Miscellany

- All taskIndices can start from the states:
  - `gap1ndex > 0`
  - `not leaf1ndex != leavesTotal and leafConnectee % leavesTotal == leavesTotal - 1`
