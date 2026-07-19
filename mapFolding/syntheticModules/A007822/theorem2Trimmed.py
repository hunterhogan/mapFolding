from mapFolding.dataBaskets import SymmetricFoldsState
from mapFolding.syntheticModules.A007822.initializeState import transitionOnGroupsOfFolds

def count(state: SymmetricFoldsState) -> SymmetricFoldsState:
    while state.leaf1ndex > 4:
        if state.leafBelow[0] == 1:
            if state.leaf1ndex > state.leavesTotal:
                state.indexLeaf = 1
                state.leafComparison[0] = 1
                state.leafConnectee = 1
                while state.leafConnectee < state.leavesTotal + 1:
                    state.indexMiniGap = state.leafBelow[state.indexLeaf]
                    state.leafComparison[state.leafConnectee] = (state.indexMiniGap - state.indexLeaf + state.leavesTotal) % state.leavesTotal
                    state.indexLeaf = state.indexMiniGap
                    state.leafConnectee += 1
                for listTuples in state.indices:
                    state.leafConnectee = 1
                    for indexLeft, indexRight in listTuples:
                        if state.leafComparison[indexLeft] != state.leafComparison[indexRight]:
                            state.leafConnectee = 0
                            break
                    state.symmetricFolds += state.leafConnectee
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
                state.indexMiniGap = state.gap1ndex
                while state.indexMiniGap < state.gap1ndexCeiling:
                    state.gapsWhere[state.gap1ndex] = state.gapsWhere[state.indexMiniGap]
                    if state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] == state.dimensionsUnconstrained:
                        state.gap1ndex += 1
                    state.countDimensionsGapped[state.gapsWhere[state.indexMiniGap]] = 0
                    state.indexMiniGap += 1
        while state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]:
            state.leaf1ndex -= 1
            state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leafBelow[state.leaf1ndex]
            state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leafAbove[state.leaf1ndex]
        state.gap1ndex -= 1
        state.leafAbove[state.leaf1ndex] = state.gapsWhere[state.gap1ndex]
        state.leafBelow[state.leaf1ndex] = state.leafBelow[state.leafAbove[state.leaf1ndex]]
        state.leafBelow[state.leafAbove[state.leaf1ndex]] = state.leaf1ndex
        state.leafAbove[state.leafBelow[state.leaf1ndex]] = state.leaf1ndex
        state.gapRangeStart[state.leaf1ndex] = state.gap1ndex
        state.leaf1ndex += 1
    else:
        state.symmetricFolds *= 2
    state.symmetricFolds = (state.symmetricFolds + 1) // 2
    return state

def doTheNeedful(state: SymmetricFoldsState) -> SymmetricFoldsState:
    state = transitionOnGroupsOfFolds(state)
    state = count(state)
    return state
