from mapFolding.dataBaskets import MapFoldingState

def count(state: MapFoldingState) -> MapFoldingState:
    while state.leaf1ndex > 4:
        if state.leafBelow[0] == 1:
            if state.leaf1ndex > state.leavesTotal:
                state.indexLeaf = 0
                leafConnectee = 0
                while leafConnectee < state.leavesTotal + 1:
                    leafNumber = int(state.leafBelow[state.indexLeaf])
                    state.leafComparison[leafConnectee] = (leafNumber - state.indexLeaf + state.leavesTotal) % state.leavesTotal
                    state.indexLeaf = leafNumber
                    leafConnectee += 1
                indexInMiddle = state.leavesTotal // 2
                state.indexMiniGap = 0
                while state.indexMiniGap < state.leavesTotal + 1:
                    ImaSymmetricFold = True
                    leafConnectee = 0
                    while leafConnectee < indexInMiddle:
                        if state.leafComparison[(state.indexMiniGap + leafConnectee) % (state.leavesTotal + 1)] != state.leafComparison[(state.indexMiniGap + state.leavesTotal - 1 - leafConnectee) % (state.leavesTotal + 1)]:
                            ImaSymmetricFold = False
                            break
                        leafConnectee += 1
                    if ImaSymmetricFold:
                        state.groupsOfFolds += 1
                    state.indexMiniGap += 1
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
        state.groupsOfFolds *= 2
    state.groupsOfFolds = (state.groupsOfFolds + 1) // 2
    return state