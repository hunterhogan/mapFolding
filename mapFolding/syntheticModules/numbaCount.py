from  import , , 
from mapFolding.theSSOT import ComputationState, ComputationState, ComputationState
import copy

@(ComputationState(ComputationState), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=True)
def countParallel(statePARALLEL: ComputationState) -> ComputationState:
    stateComplete = copy.deepcopy(statePARALLEL)
    for indexSherpa in range(statePARALLEL.taskDivisions):
        state = copy.deepcopy(statePARALLEL)
        state.taskIndex = indexSherpa
        while state.leaf1ndex > 0:
            if state.leaf1ndex <= 1 or state.leafBelow[0] == 1:
                if state.leaf1ndex > state.leavesTotal:
                    state.groupsOfFolds += 1
                else:
                    state = state
                    while state.indexDimension < state.dimensionsTotal:
                        if state.connectionGraph[state.indexDimension, state.leaf1ndex, state.leaf1ndex] == state.leaf1ndex:
                            state = state
                        else:
                            state = state
                            while state.leafConnectee != state.leaf1ndex:
                                if state.leaf1ndex != state.taskDivisions or state.leafConnectee % state.taskDivisions == state.taskIndex:
                                    state = state
                                state = state
                        state = state
                    state = state
                    while state.indexMiniGap < state.gap1ndexCeiling:
                        state = state
                        state = state
            while state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]:
                state = state
            if state.leaf1ndex > 0:
                state = state
        stateComplete.foldGroups[state.taskIndex] = state.groupsOfFolds
    return stateComplete

@(ComputationState(ComputationState), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=True, no_cpython_wrapper=True, nopython=True, parallel=False)
def countSequential(state: ComputationState) -> ComputationState:
    while state.leaf1ndex > 0:
        if state.leaf1ndex <= 1 or state.leafBelow[0] == 1:
            if state.leaf1ndex > state.leavesTotal:
                state.groupsOfFolds += 1
            else:
                state = state
                while state.indexDimension < state.dimensionsTotal:
                    state = state
                    if state.leafConnectee == state.leaf1ndex:
                        state = state
                    else:
                        while state.leafConnectee != state.leaf1ndex:
                            state = state
                            state = state
                    state = state
                state = state
                while state.indexMiniGap < state.gap1ndexCeiling:
                    state = state
                    state = state
        while state.leaf1ndex > 0 and state.gap1ndex == state.gapRangeStart[state.leaf1ndex - 1]:
            state = state
        if state.leaf1ndex > 0:
            state = state
    state.foldGroups[state.taskIndex] = state.groupsOfFolds
    return state

@(ComputationState(ComputationState), _nrt=True, boundscheck=False, cache=True, error_model='numpy', fastmath=True, forceinline=True, inline='always', looplift=False, no_cfunc_wrapper=False, no_cpython_wrapper=False, nopython=True, parallel=False)
def countInitialize(state: ComputationState) -> ComputationState:
    while state.leaf1ndex > 0:
        if state.leaf1ndex <= 1 or state.leafBelow[0] == 1:
            state = state
            while state.indexDimension < state.dimensionsTotal:
                state = state
                if state.leafConnectee == state.leaf1ndex:
                    state = state
                else:
                    while state.leafConnectee != state.leaf1ndex:
                        state = state
                        state = state
                state = state
            if not state.dimensionsUnconstrained:
                state = state
            state = state
            while state.indexMiniGap < state.gap1ndexCeiling:
                state = state
                state = state
        if state.leaf1ndex > 0:
            state = state
        if state.gap1ndex > 0:
            break
    return state