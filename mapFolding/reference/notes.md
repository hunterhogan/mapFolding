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

## Equivalent identifiers

| Lunnan | Irvine | Hunter Hogan            | diminutive    |
| ------ | ------ | ----------------------- | ------------- |
| A      | a      | leafAbove               |               |
| B      | b      | leafBelow               |               |
| C      | c      | coordinateSystem        |               |
| count  |        | countDimensionsGapped   |               |
| D      | d      | connectionGraph         |               |
| d      | dim    | dimensionsTotal         |               |
| dd     |        | dimensionsUnconstrained |               |
| delta  |        | distance                |               |
| g      |        | activeGap1ndex          |               |
| gap    |        | potentialGaps           |               |
| gapter |        | gapRangeStart           |               |
| gg     |        | gap1ndexLowerBound      |               |
| i      |        | dimension1ndex          |               |
| j      |        | indexMiniGap            |               |
| l      |        | activeLeaf1ndex         |               |
| m      |        | leaf1ndex               |               |
| m      |        | leaf1ndexConnectee      |               |
| n      |        | leavesTotal             |               |
| P      | bigP   | cumulativeProduct       |               |
| p      |        | listDimensions          |               |
|        |        | track                   | s             |
|        | mCount | foldingsTotal           |               |
|        | mod    | computationalDivisions  | taskDivisions |
|        | res    | computationalIndex      | taskIndex     |

## Potential values for variables

Assume `listDimensions` is for an implemented OEIS sequence, and
assume `listDimensions` is at most the next unknown total in the sequence.

| Type     | max(x)    | Lunnan | Hunter Hogan            |
| -------- | --------- | ------ | ----------------------- |
| 1D array | n+1       | A      | leafAbove               |
| 1D array | n+1       | B      | leafBelow               |
| 2D array | max(p)    | C      | coordinateSystem        |
| 1D array | d         | count  | countDimensionsGapped   |
| 3D array | n         | D      | connectionGraph         |
| integer  | 8*        | d      | dimensionsTotal         |
| integer  | d         | dd     | dimensionsUnconstrained |
| integer  |           | delta  | distance                |
| integer  |           | g      | activeGap1ndex          |
| 1D array | > 2*n     | gap    | potentialGaps           |
| 1D array |           | gapter | gapRangeStart           |
| integer  |           | gg     | gap1ndexLowerBound      |
| integer  | d         | i      | dimension1ndex          |
| integer  | gg-1      | j      | indexMiniGap            |
| integer  | n+1       | l      | activeLeaf1ndex         |
| integer  | n         | m      | leaf1ndex               |
| integer  | n         | m      | leaf1ndexConnectee      |
| integer  | 256*      | n      | leavesTotal             |
| 1D array | n         | P      | cumulativeProduct       |
| 1D array | 19 (2x19) | p      | listDimensions          |
| 2D array | n/a       | s      | track                   |
| integer  | ~10^17    | mCount | foldingsTotal           |

*2x2x2x2x2x2x2x2 (2x2... 8-dimensional)
