from mapFolding.dataBaskets import SymmetricFoldsState
import numpy

def transitionOnGroupsOfFolds(state: SymmetricFoldsState) -> SymmetricFoldsState:
    while state.groupsOfFolds == 0:
        if state.leaf1ndex <= 1 or state.leafBelow[0] == 1:
            if state.leaf1ndex > state.leavesTotal:
                state.indexLeaf = 1
                state.leafComparison[0] = 1
                state.leafConnectee = 1
                while state.leafConnectee < state.leavesTotal + 1:
                    state.indexMiniGap = state.leafBelow[state.indexLeaf]
                    state.leafComparison[state.leafConnectee] = (state.indexMiniGap - state.indexLeaf + state.leavesTotal) % state.leavesTotal
                    state.indexLeaf = state.indexMiniGap
                    state.leafConnectee += 1
                state.arrayGroupOfFolds = numpy.take(state.leafComparison, state.indicesArrayGroupOfFolds)
                compared = state.arrayGroupOfFolds[..., 0:state.leavesTotal // 2] == state.arrayGroupOfFolds[..., state.leavesTotal // 2:None]
                for indexRow in range(len(compared)):
                    state.groupsOfFolds += compared[indexRow].all()
            else:
                state.dimensionsUnconstrained = state.dimensionsTotal
                state.gap1ndexCeiling = state.gapRangeStart[state.leaf1ndex - 1]
                state.indexDimension = 0
                while state.indexDimension < state.dimensionsTotal:
                    state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex]
                    if state.leafConnectee == state.leaf1ndex:
                        state.dimensionsUnconstrained -= 1
                    else:
                        while state.leafConnectee != state.leaf1ndex:
                            state.gapsWhere[state.gap1ndexCeiling] = state.leafConnectee
                            if state.countDimensionsGapped[state.leafConnectee] == 0:
                                state.gap1ndexCeiling += 1
                            state.countDimensionsGapped[state.leafConnectee] += 1
                            state.leafConnectee = state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leafBelow[state.leafConnectee]]
                    state.indexDimension += 1
                if not state.dimensionsUnconstrained:
                    state.indexLeaf = 0
                    while state.indexLeaf < state.leaf1ndex:
                        state.gapsWhere[state.gap1ndexCeiling] = state.indexLeaf
                        state.gap1ndexCeiling += 1
                        state.indexLeaf += 1
                state.indexMiniGap = state.gap1ndex
                while state.indexMiniGap < state.gap1ndexCeiling:
                    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
                    if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
                        state.gap1ndex += 1
                    state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
                    state.indexMiniGap += 1
        while state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]:
            state.leaf1ndex -= 1
            state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
            state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
        if state.leaf1ndex > 0:
            state.gap1ndex -= 1
            state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
            state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
            state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
            state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
            state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
            state.leaf1ndex += 1
    state.groupsOfFolds = (state.groupsOfFolds + 1) // 2
    return state