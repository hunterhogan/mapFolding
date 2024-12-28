# JAX version

A JAX version would likely be faster than numba and it could automatically use GPU or TPU when available.
I would like to compare the speed with a functioning version.

## Notes

- int64 does not work in all environments.
- Removed `track` array to simplify the prototype.

## History

1. Building the data structures takes less than a millisecond, so I haven't optimized that step.
2. I convert the numpy.ndarray to jax.Array.
3. "pid.py" is supposed to be pure JAX but it didn't work.
4. I started over with "pider.py" as a hybrid of JAX and non-JAX.
   1. I make small changes and use the test modules to confirm the counts are correct.
   2. The hybrid module is painfully slow but the counts are correct.
5. While working on pider.py, I came up with a way to change improve parallelization, so I switched my focus to the Run Lola Run branch.
6. After returning to this branch, I decided to start over again with lunnanJAX.py

## Potential values for variables

Assume `listDimensions` is for an implemented OEIS sequence, and
assume `listDimensions` is at most the next unknown total in the sequence.

| Type | max(x) | max(array) | Identifier | Alternate Identifier |
|------|----------|-----|-------|---------------------|
| unsigned int | | n | A | leafAbove |
| unsigned int | | n | B | leafBelow |
| unsigned int | | max(p) | C | coordinateSystem |
| unsigned int |d |  | count | countDimensionsGapped |
| unsigned int | | n | D | connectionGraph |
| unsigned int |8 |  | d | dimensionsTotal |
| unsigned int |d |  | dd | unconstrainedLeaf |
| unsigned int | |  | delta | distance |
| unsigned int | |  | g | activeGap1ndex |
| unsigned int | |  | gap | potentialGaps |
| unsigned int | |  | gapter | gapRangeStart |
| unsigned int | |  | gg | gap1ndexLowerBound |
| unsigned int |d | | i | dimension1ndex |
| unsigned int |gg-1 |  | j | indexMiniGap |
| unsigned int | |  | l | activeLeaf1ndex |
| unsigned int |n | | m | leaf1ndex or leaf1ndexConnectee |
| unsigned int |256 |  | n | leavesTotal |
| unsigned int | | n | P | cumulativeProduct |
| unsigned int | | 19 | p | listDimensions |
| unsigned int | | n/a | s | track |
| unsigned int |10^17 |  | n/a | foldingsTotal |
