A = leafAbove = 0
"""Leaf above leaf m"""
B = leafBelow = 1
"""Leaf below leaf m"""
count = countDimensionsGapped = 2
"""Number of gaps available for leaf l"""
gapter = gapRangeStart = 3
"""Index of gap stack for leaf l"""

tricky = [
(activeLeaf1ndex := 0),
(activeGap1ndex := 1),
(unconstrainedLeaf := 2),
(gap1ndexLowerBound := 3),
(leaf1ndexConnectee := 4),
(taskIndex := 5),
(dimension1ndex := 6),
(foldingsSubtotal := 7),
]

COUNTindicesDynamic = len(tricky)

tricky = [
(dimensionsPlus1 := 0),
(dimensionsTotal := 1),
(leavesTotal := 2),
]

COUNTindicesStatic = len(tricky)
