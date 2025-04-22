from mapFolding.dataBaskets import Array1DElephino, Array1DLeavesTotal, Array3D, DatatypeElephino, DatatypeFoldsTotal, DatatypeLeavesTotal
from numba import jit

@jit(cache=True, error_model='numpy', fastmath=True, forceinline=True)
def count(groupsOfFolds: DatatypeFoldsTotal, gap1ndex: DatatypeElephino, gap1ndexCeiling: DatatypeElephino, indexDimension: DatatypeLeavesTotal, indexMiniGap: DatatypeElephino, leaf1ndex: DatatypeLeavesTotal, leafConnectee: DatatypeLeavesTotal, dimensionsUnconstrained: DatatypeLeavesTotal, countDimensionsGapped: Array1DLeavesTotal, gapRangeStart: Array1DElephino, gapsWhere: Array1DLeavesTotal, leafAbove: Array1DLeavesTotal, leafBelow: Array1DLeavesTotal, connectionGraph: Array3D, dimensionsTotal: DatatypeLeavesTotal, leavesTotal: DatatypeLeavesTotal) -> tuple[DatatypeFoldsTotal, DatatypeElephino, DatatypeElephino, DatatypeLeavesTotal, DatatypeElephino, DatatypeLeavesTotal, DatatypeLeavesTotal, DatatypeLeavesTotal, Array1DLeavesTotal, Array1DElephino, Array1DLeavesTotal, Array1DLeavesTotal, Array1DLeavesTotal, Array3D, DatatypeLeavesTotal, DatatypeLeavesTotal]:
    while leaf1ndex > 4:
        if leafBelow[0] == 1:
            if leaf1ndex > leavesTotal:
                groupsOfFolds += 1
            else:
                dimensionsUnconstrained = dimensionsTotal
                gap1ndexCeiling = gapRangeStart[leaf1ndex - 1]
                indexDimension = 0
                while indexDimension < dimensionsTotal:
                    leafConnectee = connectionGraph[indexDimension, leaf1ndex, leaf1ndex]
                    if leafConnectee == leaf1ndex:
                        dimensionsUnconstrained -= 1
                    else:
                        while leafConnectee != leaf1ndex:
                            gapsWhere[gap1ndexCeiling] = leafConnectee
                            if countDimensionsGapped[leafConnectee] == 0:
                                gap1ndexCeiling += 1
                            countDimensionsGapped[leafConnectee] += 1
                            leafConnectee = connectionGraph[indexDimension, leaf1ndex, leafBelow[leafConnectee]]
                    indexDimension += 1
                indexMiniGap = gap1ndex
                while indexMiniGap < gap1ndexCeiling:
                    gapsWhere[gap1ndex] = gapsWhere[indexMiniGap]
                    if countDimensionsGapped[gapsWhere[indexMiniGap]] == dimensionsUnconstrained:
                        gap1ndex += 1
                    countDimensionsGapped[gapsWhere[indexMiniGap]] = 0
                    indexMiniGap += 1
        while gap1ndex == gapRangeStart[leaf1ndex - 1]:
            leaf1ndex -= 1
            leafBelow[leafAbove[leaf1ndex]] = leafBelow[leaf1ndex]
            leafAbove[leafBelow[leaf1ndex]] = leafAbove[leaf1ndex]
        gap1ndex -= 1
        leafAbove[leaf1ndex] = gapsWhere[gap1ndex]
        leafBelow[leaf1ndex] = leafBelow[leafAbove[leaf1ndex]]
        leafBelow[leafAbove[leaf1ndex]] = leaf1ndex
        leafAbove[leafBelow[leaf1ndex]] = leaf1ndex
        gapRangeStart[leaf1ndex] = gap1ndex
        leaf1ndex += 1
    groupsOfFolds *= 2
    return (groupsOfFolds, gap1ndex, gap1ndexCeiling, indexDimension, indexMiniGap, leaf1ndex, leafConnectee, dimensionsUnconstrained, countDimensionsGapped, gapRangeStart, gapsWhere, leafAbove, leafBelow, connectionGraph, dimensionsTotal, leavesTotal)